import numpy as np
from gym_exchange import Config

class InfoGenerator():
    def __init__(self):
        self.info_index = 0

    def step(self, Self):
        self.info_index += 1
        """in an liquidation task the market_vwap ought to be
        higher, as they are not eagle to takt the liquidity,
        and can be executed at higher price."""
        '''
        step_epoch_vwap_info_dict = self.get_returned_vwap_info_dict(Self)
        '''
        step_num_hold_dict = {"Step/Current_num_hold": self.get_step_num_hold(Self)}
        step_executed_bool_dict = {"Step/Executed_bool": Self.num_left_processor.num_executed_in_last_step}
        step_cur_step_dict = {"Step/Current_step": Self.cur_step}
        step_num_left_dict = {"Step/Num_left": Self.num_left_processor.num_left}
        residual_action_dict ={"Residual_action/Quantity": Self.order_flow_generator.residual_action}
        actual_action_dict = {"Actual_action/Price":Self.wrapped_order_flow.price,
                              "Actual_action/Quantity":Self.wrapped_order_flow.quantity}
        orderbook_distance_dict = {"Orderbook/Distance": Self.orderbook_distance.get_distance(Self)}
        epoch_task_dict = {"Epoch/Num_hold": step_num_hold_dict["Step/Current_num_hold"] if Self.done else None,
                           "Epoch/Num_left": Self.num_left_processor.num_left  if Self.done else None}
        returned_info = {
            **orderbook_distance_dict,
            **actual_action_dict,
            **residual_action_dict,
            **step_num_left_dict, **step_cur_step_dict, **step_executed_bool_dict, **step_num_hold_dict,
            # **step_epoch_vwap_info_dict,
            **epoch_task_dict
            }
        return returned_info

    def get_returned_vwap_info_dict(self, Self):
        # self.vwap_estimator.update(self.exchange.executed_pairs_recoder, self.done)
        Self.vwap_estimator.update(Self.exchange.executed_pairs_recoder, Self.done)
        step_vwap_info_dict, epoch_vwap_info_dict = Self.vwap_estimator.step()
        if epoch_vwap_info_dict is None:
            return {**step_vwap_info_dict}
        else:
            return {**step_vwap_info_dict, **epoch_vwap_info_dict}

    def get_step_num_hold(self, Self):
        book = Self.exchange.order_book
        # id_lst = list(map(lambda side: list(side.order_map.keys()),[book.asks,book.bids]))
        num_in_book = 0
        for side in [book.asks,book.bids]:
            id_lst = list(side.order_map.keys())
            for id in id_lst:
                if Self.order_flow_generator.order_id_generator.initial_id <= id and id <= Self.order_flow_generator.order_id_generator.current_id:
                    num_in_book += side.get_order(id).quantity
        num_hold = Self.num_left_processor.num_left - num_in_book
        return num_hold
