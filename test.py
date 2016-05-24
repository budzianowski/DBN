class test(object):
    def __init__(self):
        self.m = 213123


def update(test):
    test.m = 213


if __name__ == '__main__':
    t = test()
    print(t.m)
    update(t)
    print(t.m)