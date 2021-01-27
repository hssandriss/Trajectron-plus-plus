def add_one():
    global a
    a += 1


if __name__ == '__main__':
    a = 0
    add_one()
    print(a)
