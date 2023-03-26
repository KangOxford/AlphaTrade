class InfoGenerator():
    def __init__(self):
        self.info_index = 0

    def step(self, Self):
        self.info_index += 1
        """in an liquidation task the market_vwap ought to be
        higher, as they are not eagle to takt the liquidity,
        and can be executed at higher price."""
        step_epoch_vwap_info_dict = self.get_returned_vwap_info_dict(Self)
        step_cur_executed_dict = {"Step/Current_executed": Self.num_left_processor.num_executed_in_last_step}
        step_cur_step_dict = {"Step/Current_step": Self.cur_step}
        step_num_left_dict = {"Step/Num_left": Self.num_left_processor.num_left}
        residual_action_dict ={"Residual_action/Quantity": Self.order_flow_generator.residual_action}
        actual_action_dict = {"Actual_action/Price":Self.wrapped_order_flow.price,
                              "Actual_action/Quantity":Self.wrapped_order_flow.quantity}
        returned_info = {
            **actual_action_dict,
            **residual_action_dict,
            **step_num_left_dict, **step_cur_step_dict, **step_cur_executed_dict,
            **step_epoch_vwap_info_dict}
        return returned_info

    def get_returned_vwap_info_dict(self, Self):
        # self.vwap_estimator.update(self.exchange.executed_pairs_recoder, self.done)
        Self.vwap_estimator.update(Self.exchange.executed_pairs_recoder, Self.done)
        step_vwap_info_dict, epoch_vwap_info_dict = Self.vwap_estimator.step()
        if epoch_vwap_info_dict is None:
            return {**step_vwap_info_dict}
        else:
            return {**step_vwap_info_dict, **epoch_vwap_info_dict}
