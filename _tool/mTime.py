
import time
fmt_day="%Y-%m-%d"
fmt_hms="%H:%M:%S"
def now_date_time():
    return time.strftime(fmt_day+" "+fmt_hms, time.localtime())
