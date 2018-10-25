from time import time
class PID:
    def __init__(self, Kp, Ki, Kd, max_integral, min_interval = 0.001, set_point = 0.0, last_time = None):
        self._Kp           = Kp
        self._Ki           = Ki
        self._Kd           = Kd
        self._min_interval = min_interval
        self._max_integral = max_integral

        self._set_point    = set_point
        self._last_time    = last_time if last_time is not None else time()
        self._p_value      = 0.0
        self._i_value      = 0.0
        self._d_value      = 0.0
        self._d_time       = 0.0
        self._d_error      = 0.0
        self._last_error   = 0.0
        self._output       = 0.0


    def update(self, cur_value, cur_time = None):
        if cur_time is None:
            cur_time = time()

        error   = self._set_point - cur_value
        d_time  = cur_time - self._last_time
        d_error = error - self._last_error

        if d_time >= self._min_interval:
            self._p_value   = error
            self._i_value   = min(max(error * d_time, -self._max_integral), self._max_integral)
            self._d_value   = d_error / d_time if d_time > 0 else 0.0
            self._output    = self._p_value * self._Kp + self._i_value * self._Ki + self._d_value * self._Kd

            self._d_time     = d_time
            self._d_error    = d_error
            self._last_time  = cur_time
            self._last_error = error

        return self._output

    def reset(self, last_time = None, set_point = 0.0):
        self._set_point    = set_point
        self._last_time    = last_time if last_time is not None else time()
        self._p_value      = 0.0
        self._i_value      = 0.0
        self._d_value      = 0.0
        self._d_time       = 0.0
        self._d_error      = 0.0
        self._last_error   = 0.0
        self._output       = 0.0

    def assign_set_point(self, set_point):
        self._set_point = set_point

    def get_set_point(self):
        return self._set_point
    
    def update_pid_factor(self, Kp, Ki, Kd):
        #print 'pid factor updated: Kp=%f, Ki=%f, Kd=%f' % (Kp, Ki, Kd)
        self._Kp           = Kp
        self._Ki           = Ki
        self._Kd           = Kd

    # def get_p_value(self):
    #     return self._p_value

    # def get_i_value(self):
    #     return self._i_value

    # def get_d_value(self):
    #     return self._d_value

    # def get_delta_time(self):
    #     return self._d_time

    # def get_delta_error(self):
    #     return self._d_error

    # def get_last_error(self):
    #     return self._last_error

    # def get_last_time(self):
    #     return self._last_time

    # def get_output(self):
    #     return self._output