import datetime
from pytz import timezone


def now_str():
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    now = now.astimezone(timezone('Asia/Tokyo'))
    return now.strftime('%Y-%m-%d %H:%M:%S')
