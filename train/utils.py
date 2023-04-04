from stable_baselines3.common.callbacks import BaseCallback


def timing(f):
    from functools import wraps
    from time import time
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te - ts))
        return result

    return wrap


def exit_after(fn):
    '''
    use as decorator to exit process if
    function takes longer than s seconds
    '''
    # from __future__ import print_function
    s = 60
    import sys
    import threading
    from time import sleep
    try:
        import thread
    except ImportError:
        import _thread as thread

    def quit_function(fn_name):
        # print to stderr, unbuffered in Python 2.
        print('{0} took too long'.format(fn_name), file=sys.stderr)
        sys.stderr.flush()  # Python 3 stderr is likely buffered.
        thread.interrupt_main()  # raises KeyboardInterrupt

    def inner(*args, **kwargs):
        timer = threading.Timer(s, quit_function, args=[fn.__name__])
        timer.start()
        try:
            result = fn(*args, **kwargs)
        finally:
            timer.cancel()
        return result

    return inner





def check_if_sorted(obs):
    print("check_if_sorted called")  # tbd
    # if not (obs == arg_sort(obs))[0].all(): return arg_sort(obs)
    # else: return obs

    assert (obs == arg_sort(obs))[0].all(), "price in obs is ont in the ascending order"

    # only check whether the price is in ascending order
    # assert (obs == -np.sort(-obs)).all(), "price in obs is ont in the ascending order"
    # todo check it


def set_sorted(obs): return arg_sort(obs)  # todo whether need to delete the check_if_sorted

def arg_sort(x):
    return x[:, x[0, :].argsort()[::-1]]


def linear_schedule(initial_value):
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress: (float)
        :return: (float)
        """
        return progress * initial_value

    return func


def biquadrate_schedule(initial_value):
    if isinstance(initial_value, str): initial_value = float(initial_value)

    def func(progress): return progress * progress * progress * progress * initial_value

    return func


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        buf_infos = self.model.env.buf_infos
        assert len(buf_infos) == 1
        # breakpoint()
        item = self.model.env.buf_infos[-1]

        # value = item.get('Step/Current_executed')
        # self.logger.record('Step/Current_executed', value)
        #
        # value = item.get('Step/Current_step')
        # self.logger.record('Step/Current_step', value)
        #
        # value = item.get('Step/Num_left')
        # self.logger.record('Step/Num_left', value)
        #
        # value = item.get('Residual_action/Quantity')
        # self.logger.record('Residual_action/Quantity', value)
        #
        # value = item.get('Actual_action/Price')
        # self.logger.record('Actual_action/Price', value)
        #
        # value = item.get('Actual_action/Quantity')
        # self.logger.record('Actual_action/Quantity', value)
        #
        # value = item.get('Orderbook/Distance')
        # self.logger.record('Orderbook/Distance', value)
        #
        # value = item.get('EpochVwap/MarketVwap')
        # self.logger.record('EpochVwap/MarketVwap', value)
        #
        # value = item.get('EpochVwap/AgentVwap')
        # self.logger.record('EpochVwap/AgentVwap', value)
        #
        # value = item.get('EpochVwap/VwapSlippage')
        # self.logger.record('EpochVwap/VwapSlippage', value)
        #
        # value = item.get('StepVwap/MarketVwap')
        # self.logger.record('StepVwap/MarketVwap', value)
        #
        # value = item.get('StepVwap/AgentVwap')
        # self.logger.record('StepVwap/AgentVwap', value)
        #
        # value = item.get('StepVwap/VwapSlippage')
        # self.logger.record('StepVwap/VwapSlippage', value)

        value = item.get('Step/Current_executed')
        self.logger.record('step/current_executed', value)

        value = item.get('Step/Current_step')
        self.logger.record('step/current_step', value)

        value = item.get('Step/Num_left')
        self.logger.record('step/num_left', value)

        value = item.get('Residual_action/Quantity')
        self.logger.record('residual_action/quantity', value)

        value = item.get('Actual_action/Price')
        self.logger.record('actual_action/price', value)

        value = item.get('Actual_action/Quantity')
        self.logger.record('actual_action/quantity', value)

        value = item.get('Orderbook/Distance')
        self.logger.record('orderbook/distance', value)

        value = item.get('EpochVwap/MarketVwap')
        self.logger.record('epochvwap/marketvwap', value)

        value = item.get('EpochVwap/AgentVwap')
        self.logger.record('epochvwap/agentvwap', value)

        value = item.get('EpochVwap/VwapSlippage')
        self.logger.record('epochvwap/vwapslippage', value)

        value = item.get('StepVwap/MarketVwap')
        self.logger.record('stepvwap/marketvwap', value)

        value = item.get('StepVwap/AgentVwap')
        self.logger.record('stepvwap/agentvwap', value)

        value = item.get('StepVwap/VwapSlippage')
        self.logger.record('stepvwap/vwapslippage', value)

        # try:
        #     value = item.get('penalty_delta')
        #     self.logger.record('env/penalty_delta', value)
        # except:
        #     pass
        return True



if __name__ == "__main__":
    pairs = [[123, 1], [133324, 1], [132312, 3]]  ##
    # series = observation[0]
