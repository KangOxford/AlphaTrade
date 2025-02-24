from functools import partial, partialmethod
from typing import OrderedDict
from jax import numpy as jnp
import jax
import numpy as np
import random
import time
import timeit

import sys
# ******** INSERT PATH HERE ********
sys.path.append('/home/duser/AlphaTrade/')
import gymnax_exchange
import gymnax_exchange.jaxob.JaxOrderBookArrays as job
import gymnax_exchange.utils.utils as utils



class TestLimitOrderBookSimulator:
    def __init__(self):
        self.cfg=job.Configuration()


    def test_add_order_to_full_book(self):
        book=utils.create_init_book(self.cfg,order_capacity=100,trade_capacity=100,percent_fill=1)
        mdict,marray=utils.create_rand_message(type='limit',side='bid')
        book_out=job.cond_type_side(self.cfg,book,mdict)
        assert book_out==book

    def setup_method(self):
        self.simulator = None

    def test_add_order(self):
        pass

    def test_cancel_order(self):
        pass

    def test_match_orders(self):
        pass

    def test_get_order_book(self):
        pass


class SpeedExperimentsCore:
    def __init__(self):
        self.simulator = None
        self.cfg = job.Configuration()
        self.key = jax.random.PRNGKey(self.cfg.seed)

    def run_speed_tests_and_plot(self, n_orders, book_capacities, n_samples=100,vmap=False,n_vmap=1000,suffix=""):
        add_order_times = []
        match_order_times = []
        cancel_order_times = []

        for booksize in book_capacities:
            mean_time, lower_bound, upper_bound = self.test_speed_add_order(n_orders, booksize, rand_orders=True, n_samples=n_samples,vmap=vmap,n_vmap=n_vmap)
            add_order_times.append((mean_time, lower_bound, upper_bound))

            mean_time, lower_bound, upper_bound = self.test_speed_match_orders(n_orders, booksize, find_order=True, match_order=True, n_samples=n_samples,vmap=vmap,n_vmap=n_vmap)
            match_order_times.append((mean_time, lower_bound, upper_bound))

            mean_time, lower_bound, upper_bound = self.test_speed_cancel_order(n_orders, booksize, n_samples=n_samples,vmap=vmap,n_vmap=n_vmap)
            cancel_order_times.append((mean_time, lower_bound, upper_bound))

        self.plot_results(suffix,book_capacities, add_order_times, match_order_times, cancel_order_times)

    def plot_results(self,suffix, x, add_order_times, match_order_times, cancel_order_times):
        import matplotlib.pyplot as plt

        add_order_means, add_order_lowers, add_order_uppers = zip(*add_order_times)
        match_order_means, match_order_lowers, match_order_uppers = zip(*match_order_times)
        cancel_order_means, cancel_order_lowers, cancel_order_uppers = zip(*cancel_order_times)

        plt.figure(figsize=(10, 6))
        plt.plot(x, add_order_means, label='Add Order')
        plt.fill_between(x, add_order_lowers, add_order_uppers, alpha=0.2)
        
        plt.plot(x, match_order_means, label='Match Order')
        plt.fill_between(x, match_order_lowers, match_order_uppers, alpha=0.2)
        
        plt.plot(x, cancel_order_means, label='Cancel Order')
        plt.fill_between(x, cancel_order_lowers, cancel_order_uppers, alpha=0.2)
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Number of Orders')
        plt.ylabel('Time (seconds)')
        plt.title('Speed Test Results')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.savefig(f'speed_test_results_{suffix}.png')
        plt.show()

    def bootstrap_confidence_interval(self, data, num_samples=1000, confidence_level=0.99):
        sample_means = []
        n = len(data)
        for _ in range(num_samples):
            sample = [random.choice(data) for _ in range(n)]
            sample_means.append(np.mean(sample))
        lower_bound = np.percentile(sample_means, (1 - confidence_level) / 2 * 100)
        upper_bound = np.percentile(sample_means, (1 + confidence_level) / 2 * 100)
        return lower_bound, upper_bound

    def test_speed_add_order(self,n_orders,booksize,rand_orders=True,n_samples=100,vmap=False,n_vmap=1000):
        times = []
        if not vmap:
            n_vmap=1
            for _ in range(n_samples):
                asks,bids,trades=utils.create_init_book(self.cfg,order_capacity=booksize,trade_capacity=booksize)
                orders=[]
                for _ in range(n_orders):
                    mdict,marray=utils.create_rand_message(type='limit',side='bid')
                    orders.append(mdict)
                if rand_orders:
                    start_time = time.time()
                    for order in orders:
                        bids=job.add_order(bids,order)
                    end_time = time.time()
                    times.append((end_time - start_time)/n_orders/n_vmap)
                else:
                    start_time = time.time()
                    for order in orders:
                        out=job.add_order(bids,orders[0])
                    end_time = time.time()
                    times.append((end_time - start_time)/n_orders/n_vmap)
        else:
            add_vmap=jax.jit(jax.vmap(job.add_order,in_axes=(0,0)))
            for i in range(n_samples):
                print(f"{i}",end="\r")
                vbids = []
                for _ in range(n_vmap):
                    asks, bids, trades = utils.create_init_book(self.cfg,order_capacity=booksize, trade_capacity=booksize)
                    vbids.append(bids)
                vbids = jnp.array(vbids)
                order_dicts = [utils.create_rand_message(type="limit", side='bid')[0] for _ in vbids]
                order_dicts = {key: jnp.array([d[key] for d in order_dicts]) for key in order_dicts[0].keys()}
                start_time = time.time()
                for _ in range(n_orders):
                    out=add_vmap(vbids,order_dicts)
                end_time = time.time()
                times.append((end_time - start_time)/n_orders/n_vmap)


        # Remove the first time due to compilation overhead.
        times=times[1:]
        lower_bound, upper_bound = self.bootstrap_confidence_interval(times)
        mean_time = np.mean(times)
        print(f"Mean add time per order for {n_orders} order in {n_vmap} books of size {booksize}: {mean_time} seconds")
        print(f"99% confidence interval for adding {n_orders} orders in a book of size {booksize}: [{lower_bound}, {upper_bound}] seconds")
        return mean_time, lower_bound, upper_bound

    def test_speed_match_orders(self,n_orders,booksize,find_order=True,match_order=True,n_samples=100,vmap=False,n_vmap=1000):
        times = []
        if not vmap:
            n_vmap=1
            for _ in range(n_samples):
                asks,bids,trades=utils.create_init_book(self.cfg,order_capacity=booksize,trade_capacity=booksize)
                orders=[]
                for _ in range(n_orders):
                    mdict,marray=utils.get_random_aggressive_order(bids,side='bid')
                    orders.append(mdict)
                
                if find_order and match_order:
                    start_time = time.time()
                    for order in orders:
                        out,qtm,price,trade=job._match_against_bid_orders(self.cfg,bids,order["quantity"],order["price"],trades,order["orderid"],order["time"],order["time_ns"],order["orderid"],job.cst.BidAskSide.BID.value)
                    end_time = time.time()
                    times.append(end_time - start_time)
                elif find_order:
                    start_time = time.time()
                    for order in orders:
                        out=job._get_top_bid_order_idx(self.cfg,bids)
                    end_time = time.time()
                    times.append(end_time - start_time)
                elif match_order:
                    matchtuples=[]
                    for order in orders:
                        idx=job._get_top_bid_order_idx(self.cfg,bids)
                        matchtuples.append((idx,bids,order["quantity"],order["price"],trades,order["orderid"],order["time"],order["time_ns"],order["orderid"],job.cst.BidAskSide.BID.value))
                    start_time = time.time()
                    for matchtuple in matchtuples:
                        out=job.match_order(matchtuple)
                    end_time = time.time()
                    times.append((end_time - start_time)/n_orders/n_vmap)
        else:
            match_vmap=jax.jit(jax.vmap(partial(job._match_against_bid_orders, self.cfg),in_axes=(0,0,0,0,0,0,0,0,None)))
            for i in range(n_samples):
                print(f"{i}",end="\r")
                vbids,vtrades = [],[]
                for _ in range(n_vmap):
                    asks, bids, trades = utils.create_init_book(self.cfg,order_capacity=booksize, trade_capacity=booksize)
                    vbids.append(bids)
                    vtrades.append(trades)
                vbids = jnp.array(vbids)
                vtrades = jnp.array(vtrades)

                order_dicts = [utils.get_random_aggressive_order(bid, side='bid')[0] for bid in vbids]
                order_dicts = {key: jnp.array([d[key] for d in order_dicts]) for key in order_dicts[0].keys()}
                start_time = time.time()
                for _ in range(n_orders):
                    out = match_vmap(vbids, order_dicts["quantity"],order_dicts["price"],vtrades,order_dicts["orderid"],order_dicts["time"],order_dicts["time_ns"],order_dicts["orderid"],job.cst.BidAskSide.BID.value)
                end_time = time.time()
                times.append((end_time - start_time)/n_orders/n_vmap)

        # Remove the first time due to compilation overhead.
        times=times[1:]
        lower_bound, upper_bound = self.bootstrap_confidence_interval(times)
        mean_time = np.mean(times)
        print(f"Mean time for matching {n_orders} orders in a book of size {booksize}: {mean_time} seconds")
        print(f"99% confidence interval for matching {n_orders} orders in a book of size {booksize}: [{lower_bound}, {upper_bound}] seconds")
        return mean_time, lower_bound, upper_bound

    def test_speed_cancel_order(self,n_orders,booksize,n_samples=100,vmap=False,n_vmap=1000):
        times = []
        if not vmap:
            n_vmap=1
            for _ in range(n_samples):
                asks, bids, trades = utils.create_init_book(self.cfg,order_capacity=booksize, trade_capacity=booksize)
                order_dict,_ = utils.get_random_order_to_cancel(bids,side='bid')
                start_time = time.time()
                for _ in range(n_orders):
                    out=job.cancel_order(self.cfg, self.key,bids,order_dict)
                end_time = time.time()
                times.append((end_time - start_time)/n_orders/n_vmap)
        else:
            cancel_vmap=jax.jit(jax.vmap(partial(job.cancel_order, self.cfg, self.key),in_axes=(0,0)))
            for i in range(n_samples):
                print(f"{i}",end="\r")
                vbids = []
                for _ in range(n_vmap):
                    asks, bids, trades = utils.create_init_book(self.cfg,order_capacity=booksize, trade_capacity=booksize)
                    vbids.append(bids)
                vbids = jnp.array(vbids)
                order_dicts = [utils.get_random_order_to_cancel(bid, side='bid')[0] for bid in vbids]
                order_dicts = {key: jnp.array([d[key] for d in order_dicts]) for key in order_dicts[0].keys()}
                start_time = time.time()
                for _ in range(n_orders):
                    out = cancel_vmap(vbids, order_dicts)
                end_time = time.time()
                times.append((end_time - start_time)/n_orders/n_vmap)



        # Remove the first time due to compilation overhead.
        times = times[1:]
        lower_bound, upper_bound = self.bootstrap_confidence_interval(times)
        mean_time = np.mean(times)
        print(f"Mean cancelation time per order for {n_orders} orders in {n_vmap} books of size {booksize}: {mean_time} seconds")
        print(f"99% confidence interval for canceling {n_orders} orders in a book of size {booksize}: [{lower_bound}, {upper_bound}] seconds")
        return mean_time, lower_bound, upper_bound



if __name__ == "__main__":
    tester = TestLimitOrderBookSimulator()

    # tester.test_add_order_to_full_book()

    speed_tester = SpeedExperimentsCore()
    # speed_tester.test_speed_cancel_order(1000,100,vmap=False,n_vmap=1000,n_samples=100)
    # speed_tester.test_speed_cancel_order(1000,  100,vmap=True,n_vmap=1000,n_samples=10)
    speed_tester.run_speed_tests_and_plot(1000, [10,30, 100,300,1000,3000],vmap=False,n_samples=100,suffix="no_vmap")
    speed_tester.run_speed_tests_and_plot(1000,  [10,30, 100,300,1000,3000],vmap=True,n_samples=25,suffix="vmap")

    # speed_tester.test_speed_add_order(1000,100,rand_orders=False)
    # speed_tester.test_speed_add_order(1000,100,rand_orders=False,vmap=True,n_vmap=1000,n_samples=10) 

    # speed_tester.test_speed_match_orders(1000,100,find_order=True,match_order=False)

