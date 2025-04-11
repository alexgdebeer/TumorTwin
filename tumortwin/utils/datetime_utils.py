from datetime import datetime, timedelta


def days_since_first(date: datetime, first: datetime) -> float:
    """
    Calculates the number of days since the first treatment.

    Args:
        date (datetime): The current date.
        first (datetime): The date of the first treatment.

    Returns:
        float: The number of days since the first treatment.
    """
    return timedelta_to_days(date - first)


def timedelta_to_days(td: timedelta) -> float:
    """
    Convert a timedelta object to a duration in days.

    Args:
        td (timedelta): The timedelta to convert.

    Returns:
        float: The total duration in days.
    """
    return td.total_seconds() / (60.0 * 60.0 * 24.0)


def daterange(t_initial: datetime, t_final: datetime, dt: timedelta):
    """
    Generate a range of dates between two times with a fixed interval.

    Args:
        t_initial (datetime): The start of the date range.
        t_final (datetime): The end of the date range.
        dt (timedelta): The interval between successive dates.

    Returns:
        list[datetime]: A list of datetime objects representing the range.
    """
    ts = [t_initial]
    while ts[-1] < t_final:
        ts.append(ts[-1] + dt)
    return ts
