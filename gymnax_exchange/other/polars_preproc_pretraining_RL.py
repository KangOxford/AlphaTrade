from decimal import Decimal
import os
import glob
import polars as pl
import polars.selectors as cs
import numpy as np
import datetime
from tqdm import tqdm



def extract_date(fname):
    date_str = fname.split("_", maxsplit=2)[1]
    date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    return date


def load_data(
    message_files,
    *,
    hide_old_ids=True,
    renumber_ids=True,
    price_norm_strategy="eod",
    padding=100,
    start_date=None,
    end_date=None,
):
    if isinstance(message_files, str):
        message_files = sorted(glob.glob(message_files))
    elif isinstance(message_files, list):
        # print("List of message files:", message_files)
        # message_files = sorted(message_files)
        all_msg_files = []
        for m_f in message_files:
            all_msg_files.extend(glob.glob(m_f))
        message_files = sorted(all_msg_files)
    else:
        raise ValueError("message_files must be a string or list of strings")

    if start_date is not None:
        start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        # filter to files after given start_date
        message_files = [m_f for m_f in message_files if extract_date(m_f) >= start_date]
    if end_date is not None:
        end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        # filter to files before given end_date
        message_files = [m_f for m_f in message_files if extract_date(m_f) <= end_date]

    print("Loading message files:")
    print("\n".join(message_files))

    queries = []
    for m_f in tqdm(message_files):
        if not os.path.exists(m_f):
            raise FileNotFoundError(f"File not found: {m_f}")
        fname = os.path.basename(m_f)
        stock, date = fname.split("_", maxsplit=2)[:2]
        queries.append(
            _process_file(
                m_f,
                hide_old_ids=hide_old_ids,
                renumber_ids=renumber_ids,
                padding=padding
            ).select(
                pl.lit(stock).cast(pl.Categorical).alias("stock"),
                pl.lit(date).cast(pl.Date).alias("date"),
                pl.all(),
            )
        )
    print("finished loading message files")
    df = pl.concat(queries, how="vertical")
    print("finished concat")
    if price_norm_strategy is not None:
        df = _normalize_price(df, strategy=price_norm_strategy)
    print("normalized prices")
    # print("df height", df.height)
    return df


def _process_file(m_f, *, hide_old_ids=True, renumber_ids=True, padding=100):
    messages = pl.scan_csv(
        m_f,
        has_header=False,
        truncate_ragged_lines=True,
        schema={
            "time": pl.Decimal(precision=14, scale=9),
            "event": pl.UInt8,
            "order_id": pl.Int32,
            "size": pl.UInt32,
            "price": pl.UInt32,
            "direction": pl.Int8,
            # "exchange": pl.Utf8, # not used --> mostly null, ignore
        }
    ).with_columns(
        direction = (pl.col("direction") == 1),
        price = pl.col("price") // 100,
    ).with_columns(
        delta_time = (pl.col("time") - pl.col("time").shift(1)),
    # skip first message to align with prior book
    ).slice(1, None)

    # Convert execution orders (type 4) to limit orders (type 1) with flipped direction
    messages = messages.with_columns(
        direction = pl.when(pl.col("event") == 4).then(~pl.col("direction")).otherwise(pl.col("direction")),
        event = pl.when(pl.col("event") == 4).then(pl.lit(1)).otherwise(pl.col("event"))
    )

    # Convert delete orders (type 3) to cancel orders (type 2)
    messages = messages.with_columns(
        event = pl.when(pl.col("event") == 3).then(pl.lit(2)).otherwise(pl.col("event"))
    )

    if hide_old_ids:
        messages = _set_old_ids_to(messages, 0)
    if renumber_ids:
        messages = _renumber_ids(messages)

    book_str = m_f.replace("message", "orderbook")
    book = pl.scan_csv(
        book_str,
        has_header=False,
    )
    book_schema = book.collect_schema()
    n_cols = len(book_schema)
    # replace sentinel values with None and cast to Int32
    book = book.select([
        pl.col(c).replace({"-9999999999": None, "9999999999": None}).cast(pl.Int32).alias(c)
        for c in book_schema.names()
    ])

    # book = book.cast(pl.UInt32)
    # n_rows = book.select(pl.len()).collect().item()
    n_levels = n_cols // 4
    book_cols = [c for l in range(1, n_levels+1) for c in [f"p_a_{l}", f"v_a_{l}", f"p_b_{l}", f"v_b_{l}"]]
    rename_dict = {f"column_{i+1}": c for i, c in enumerate(book_cols)}
    book = book.rename(rename_dict)

    # divide all book price (columns starting with "p_") by 100 (min tick size)
    book = book.with_columns([
        pl.col(c) // 100 for c in rename_dict.values() if c.startswith("p_")
    ])

    # drop last book to align prior book with following message
    # book = book.slice(0, n_rows - 1)
    zero_row = messages.clear(1) #pl.DataFrame({col: [0] for col in messages.columns})

    # Add the zero row to the DataFrame
    # messages = messages.vstack(zero_row)
    messages = pl.concat([messages, zero_row])

    df = pl.concat([messages, book], how="horizontal")#.collect()
    # ignore hidden orders and trade pauses
    df = df.filter(pl.col("event") <= 4)

    # best price move in next book state (next after message)
    delta_ask = pl.col("p_a_1").diff(-1).alias("delta_ask")
    delta_bid = pl.col("p_b_1").diff(-1).alias("delta_bid")
    # sum columns because only one can be non-zero
    df = df.with_columns(delta_price=delta_ask + delta_bid)

    # zero row for start of day (message + book both 0)
    zero_row = df.clear(1)
    df = pl.concat([zero_row, df])

    # fill potential None values in prices and delta_price with 0
    df = df.fill_null(strategy="zero")

    height = df.select(pl.len()).collect().item()# + 1

    # print(height, height % padding)
    # if height % padding == 1:
    #     df = df.slice(0, height - 1)
    # el
    if height % padding != 0:
        num_to_add = padding - (height % padding)
        # print(num_to_add)
        to_add = df.tail(1)

        # padding repeats the last book state and sets message to None (-> later 0)
        schema = to_add.collect_schema()
        col_names = schema.names()
        to_add = to_add.with_columns(
            pl.lit(None).cast(schema[col]).alias(col)
            for col in col_names
            if (
                (col not in ["stock", "date"])
                and (not col.startswith("p_"))
                and (not col.startswith("v_"))
            )
        )

        df = pl.concat([df] + [to_add] * num_to_add, rechunk=True)
    # print("new height", df.select(pl.len()).collect().item())

    # enumerate messages of the day
    df = df.with_columns(
        msg_num = pl.arange(pl.len())
    )

    # NOTE: leave materialization to the caller
    df = df.collect()
    # print(height, df.height)
    return df


def _set_old_ids_to(df, value=0):
    # Step 1: Find order_ids that have at least one event == 1
    order_ids_with_event_1 = (
        df.filter(pl.col("event") == 1)
        .select("order_id")
        .unique()
    )
    # Step 2: Left-join the original dataframe with the filtered order_ids
    df = df.join(order_ids_with_event_1, on="order_id", how="left", suffix="_exists", maintain_order="left", coalesce=False)
    # Step 3: If order_id is missing in the joined frame, replace it with `value`
    df = df.with_columns(
        pl.when(pl.col("order_id_exists").is_null())
        .then(value)
        .otherwise(pl.col("order_id"))
        .alias("order_id")
    ).drop("order_id_exists")
    return df


def _renumber_ids(df):
    # Step 1: Find all unique order_ids that are not 0
    # NOTE: don't replace 0s, as these are special vals for unknown previous day IDs
    replace_ids = df.select(
        pl.col("order_id").filter(pl.col("order_id") != 0).unique(maintain_order=True),
    )
    # Step 2: Create a replacement column with new order ids
    # new column with new order ids (newly enumerated)
    replace_ids = replace_ids.with_columns(
        pl.arange(1, pl.len() + 1).alias("new_id")
    )
    # Step 3: replace old order ids with new order ids
    df = df.join(
        replace_ids, on="order_id", how="left", suffix="_new", maintain_order="left", coalesce=True
    ).drop("order_id").rename({"new_id": "order_id"})
    return df


def _normalize_price(df, strategy="eod"):
    if strategy == "eod":
        df = df.with_columns(
            first_start_price = ((pl.col("p_a_1") + pl.col("p_b_1")) // 2).first().over("date")
        )
        # end-of-day mid-price by date
        last_price = df.group_by("date").agg(
            ((pl.last("p_a_1") + pl.last("p_b_1")) // 2).alias("price_ref"),
            pl.col("first_start_price").last()
        ).with_columns(
            pl.col("price_ref").shift(1).fill_null(pl.col("first_start_price"))
        ).drop("first_start_price")
        df = df.drop("first_start_price")

        # join last price with original dataframe
        df = df.join(last_price, on="date", how="left", maintain_order="left", coalesce=True)
    elif strategy == "sod":
        df = df.with_columns(
            price_ref = ((pl.col("p_a_1") + pl.col("p_b_1")) // 2).first().over("date")
        )
    else:
        raise ValueError(f"Unknown normalization strategy: {strategy}")

    # calculate normalized price: subtract reference price from all price columns
    df = df.with_columns(cs.matches('^p_') - pl.col("price_ref"))
    df = df.with_columns(pl.col("price") - pl.col("price_ref"))
    # drop reference price
    # df = df.drop("price_ref")
    return df


def reorder_cols(df, start_cols=None, all_cols=None):
    if start_cols is None:
        start_cols = [
            "stock",
            "date",
            "event",
            "direction",
            "order_id",
            "price",
            "size",
            "delta_time",
            "delta_price",
            "price_ref",
            "time",
        ]
    if all_cols is None:
        all_cols = df.columns
    cols = [c for c in all_cols if c not in start_cols]
    start_cols.extend(cols)
    return df.select(start_cols)


def tok_preproc(df: pl.DataFrame) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    # (1) split timestamps into seconds and nanoseconds
    print("handle time")
    df_out = df.with_columns(
        time_s = pl.col("time").floor().cast(pl.Int32),
        delta_time_s = pl.col("delta_time").floor().cast(pl.Int32)
    ).with_columns(
        time_ns = (((pl.col("time") - pl.col("time_s"))) \
                .cast(pl.Decimal(precision=21, scale=9)) * 1_000_000_000) \
                .cast(pl.Int32),
        delta_time_ns = (((pl.col("delta_time") - pl.col("delta_time_s"))) \
                .cast(pl.Decimal(precision=21, scale=9)) * 1_000_000_000) \
                .cast(pl.Int32),
    ).with_columns(
    # (2) combine columns event and direction into a single column event_dir as crossproduct
        event_dir = (pl.col("direction").cast(pl.UInt8) * 4) + pl.col("event").cast(pl.UInt8),
    ).drop("delta_time", "time", "event", "direction", "price_ref")

    # print("collecting schema")
    # schema = df_out.collect_schema()
    # col_names = schema.names()

    # Do this per file (day) instead in _process_file
    # # (3) prepend start of day message
    # print("prepending sod")
    # sod_message = df_out.filter(pl.col("msg_num") == 0)
    # # set all message columns except stock and date to None, keep book columns
    # sod_message = sod_message.with_columns(
    #     pl.lit(None).cast(schema[col]).alias(col)
    #     for col in col_names
    #     if (
    #         (col not in ["stock", "date"])
    #         and (not col.startswith("p_"))
    #         and (not col.startswith("v_"))
    #     )
    # )
    # print("sorting sod")
    # # sort in the sod messages at the start of each day
    # df_out = pl.concat(
    #     [sod_message, df_out], how="vertical"
    # ).sort(["date", "msg_num"])

    print("removing columns")
    # remove columns not needed for tokenization
    df_out = df_out.drop("msg_num", "stock", "date").fill_null(strategy="zero")
    # materialize if lazy up to here
    # print("rematerializing")
    # df_out = df_out.collect()
    print("reordering cols")
    df_out = reorder_cols(
        df_out,
        [
            "event_dir",
            "order_id",
            "price",
            "size",
            "delta_time_s",
            "delta_time_ns",
            "delta_price",
            "time_s",
            "time_ns"
        ],
    )

    print(df_out)
    print(df_out.columns)

    # split back into message, and orderbook data
    msg_cols = df_out.columns[:7]
    book_cols = df_out.columns[7:]
    message = df_out.select(msg_cols).cast(pl.Int32)
    orderbook = df_out.select(book_cols)
    return msg_cols, book_cols, message.to_numpy(), orderbook.to_numpy()


def processed_messages_to_lobster(
    msg_in: np.ndarray,
    p_ref: np.ndarray,
    t_start: np.ndarray,
) -> pl.DataFrame:
    # add 3 new message columns: p_ref (2), t_start (1)
    msg = np.empty(msg_in.shape[:-1] + (msg_in.shape[-1] + 3,), dtype=msg_in.dtype)
    msg[..., :-3] = msg_in

    # replace price changes (not needed) with batch index
    msg[..., -4] = np.arange(msg.shape[0])[..., None]
    # append reference prices and start_times as new columns
    # msg = np.concatenate([msg, np.zeros(msg.shape[:-1] + (3,), dtype=np.int32)], axis=-1)
    msg[..., -3] = p_ref[..., None]
    msg[..., -2:] = t_start[:, None]

    df = pl.DataFrame(
        # drop price changes --> not needed for lobster format messages
        msg.reshape(-1, msg.shape[-1]),
        schema=[
            'event_dir',
            'order_id',
            'price',
            'size',
            'delta_time_s',
            'delta_time_ns',
            'batch',
            'p_ref',
            't_start_s',
            't_start_ns',
        ]
    )
    print(df)
    df = df.with_columns(
        # split event_dir into event and dir. direction from {0,1} to {-1,1}
        direction = (pl.col('event_dir') // 5) * 2 - 1,
        event = pl.col('event_dir') - (pl.col('event_dir') // 5) * 4,
        # add reference price and multiply by 100 to LOBSTER price scale
        price = (pl.col('price') + pl.col('p_ref')) * 100,
        # combine seconds and nanoseconds to string timestamp
        delta_time = (
            pl.col("delta_time_s").cast(pl.Utf8) + "." +
            pl.col("delta_time_ns").cast(pl.Utf8).str.zfill(9)
        # should be sufficient for valid tokens:
        # ).cast(pl.Decimal(precision=15, scale=9)),
        # allow for generation errors --> timestamps too long:
        ).cast(pl.Decimal(precision=30, scale=9)),
        start_time = (
            pl.col("t_start_s").cast(pl.Utf8) + "." +
            pl.col("t_start_ns").cast(pl.Utf8).str.zfill(9)
        # should be sufficient for valid tokens:
        # ).cast(pl.Decimal(precision=15, scale=9))
        # allow for generation errors --> timestamps too long:
        ).cast(pl.Decimal(precision=15, scale=9))
    )
    # add up time deltas to timestamps
    df = df.with_columns(time = pl.col("delta_time").cum_sum() + pl.col("start_time"))
    # bring columns into lobster order
    df = df.select(["time", "event", "order_id", "size", "price", "direction"])

    # check this works
    return df


if __name__ == "__main__":
    # use argparse to parse command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("message_files", nargs="+", help="List of message files to load")
    parser.add_argument("--save_dir", default="data", help="Path to save the data")
    parser.add_argument("--keep_old_ids", action="store_true", help="Do not hide old order ids")
    parser.add_argument("--skip_renumber_ids", action="store_true", help="Do not renumber order ids")
    parser.add_argument("--price_norm_strategy", default="eod", help="Price normalization strategy")
    parser.add_argument("--day_padding", default=100, type=int)
    parser.add_argument("--start_date", default=None, type=str)
    parser.add_argument("--end_date", default=None, type=str)

    args = parser.parse_args()

    if args.start_date is not None:
        print("Starting date is:", args.start_date)
    if args.end_date is not None:
        print("Ending date is:", args.end_date)

    print("Using price normalization strategy:", args.price_norm_strategy)

    # load data
    df = load_data(
        args.message_files,
        hide_old_ids=(not args.keep_old_ids),
        renumber_ids=(not args.skip_renumber_ids),
        price_norm_strategy=args.price_norm_strategy,
        padding=args.day_padding,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print("starting preproc")
    # preprocess data and convert to numpy arrays
    msg_cols, book_cols, message, orderbook = tok_preproc(df)
    print("message columns:", msg_cols)
    print("orderbook columns:", book_cols)
    # save to npy files
    file_postfix = f"_{args.day_padding}_{args.price_norm_strategy}"
    if args.start_date is not None:
        file_postfix += f"_sd-{args.start_date}"
    if args.end_date is not None:
        file_postfix += f"_ed-{args.end_date}"
    os.makedirs(args.save_dir, exist_ok=True)
    np.save(args.save_dir + f"/messages{file_postfix}.npy", message)
    np.save(args.save_dir + f"/orderbooks{file_postfix}.npy", orderbook)
