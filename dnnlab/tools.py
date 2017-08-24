import datetime


def base36encode(number):
    """encode an integer with base 36
    """

    if not isinstance(number, (int)):
        raise TypeError('number must be an integer')
    if number < 0:
        raise ValueError('number must be positive')

    alphabet, base36 = ['0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', '']

    while number:
        number, i = divmod(number, 36)
        base36 = alphabet[i] + base36

    return base36 or alphabet[0]


def base36decode(number):
    return int(number, 36)


def delta_data(start=datetime.date(2017, 8, 1)):
    """return the number of days passed from `start`
    """

    today = datetime.date.today()
    delta = today - start
    return delta.days
