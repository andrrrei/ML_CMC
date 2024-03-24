from typing import List


def hello(name = '') -> str:
    if name != '':
        return f'Hello, {name}!'
    else:
        return 'Hello!'


def int_to_roman(num: int) -> str:
    digits = [0] * 4
    i = 1
    while num > 0:
        digits[-i] = num % 10
        i += 1
        num //= 10
    rom = 'MCXI'
    rom5 = 'DLV'
    res = ''
    for i in range(len(digits)):
            if digits[i] == 4:
                res += rom[i] + rom5[i - 1]
            if digits[i] == 9:
                res += rom[i] + rom[i - 1]
            if digits[i] in [5, 6, 7, 8]:
                res += rom5[i - 1] + rom[i] * (digits[i] - 5)
            if digits[i] in [1, 2, 3]:
                res += rom[i] * digits[i]
    return res



def longest_common_prefix(strs_input: List[str]) -> str:
    if len(strs_input) > 0:
        res = strs_input[0].strip()
        for i in range(1, len(strs_input)):
            strs_input[i] = strs_input[i].strip()
            if len(strs_input[i]) < len(res):
                res = strs_input[i]
            else:
                for j in range(len(res)):
                    if res[j] != strs_input[i][j]:
                        res = res[:j]
                        break
        return res
    else:
        return ''


def primes() -> int:
    i = 2
    while(1):
        flag = True
        for j in range(2, int(i ** 0.5 + 1)):
            if i % j == 0:
                flag = False
                break
        if flag:
            yield i
        i += 1


class BankCard:
    
    def __init__(self, total_sum: int, balance_limit = 10**5):
        self.total_sum = total_sum
        self.balance_limit = balance_limit

    def spend_money(self, sum_spent: int):
        try:
            if sum_spent > self.total_sum:
                raise ValueError
            else:
                self.total_sum -= sum_spent
                print(f'You spent {sum_spent} dollars')
        except ValueError:
            print('Not enough money to spend {sum_spent} dollars.')

    def __call__(self, sum_spent: int):
        self.spend_money(sum_spent)

    def __repr__(self):
        return "To learn the balance call balance."
    @property
    def balance(self):
        try:
            if self.balance_limit == 0:
                raise ValueError
            else:
                self.balance_limit -= 1
                return self.total_sum
        except ValueError:
            print(f"Balance check limits exceeded.")

    def put(self, sum_put):
        self.total_sum += sum_put
        print(f'You put {sum_put} dollars.')
    

    def __add__(self, card2):
        return BankCard(self.total_sum + card2.total_sum, max(self.balance_limit, card2.balance_limit))
    
