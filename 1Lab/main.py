# Первая лаба
# Номер варика 1 + (17*31+7*0+26)%20 = 14
# d = 7
# Функция Леви

def f(x:list[float|int], d) -> float|int:
    return 0.5 * sum(x[i] ** 4 - 16 * x[i] ** 2 + 5 * x[i] for i in range(d))
