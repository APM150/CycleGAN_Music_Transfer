import os
import csv

class Logwriter:

    def __init__(self, directory: str):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        self.pagewriters = []

    def get_page_writer(self, page: str):
        pagewriter = Pagewriter(self, page)
        self.pagewriters.append(pagewriter)
        return pagewriter


class Pagewriter:

    def __init__(self, writer: Logwriter, page: str):
        self.writer = writer
        self.page = page

    def write(self, fig: str, x: int, y: float):
        path = os.path.join(self.writer.directory, self.page + '_' + fig + '.csv')
        with open(path, 'a') as f:
            csv.writer(f).writerow([x, y])


if __name__ == '__main__':
    logwriter = Logwriter('exp/jun28test')
    pagewriter = logwriter.get_page_writer('test1')
    for i in range(10):
        pagewriter.write('fig1', i, i * 3.15)
