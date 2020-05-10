import requests
from os.path import expanduser
import os


class ProgressBar(object):

    def __init__(self, title,
                 count=0.0,
                 run_status=None,
                 fin_status=None,
                 total=100.0,
                 unit='', sep='/',
                 chunk_size=1.0):
        super(ProgressBar, self).__init__()
        if total is not None:
            self.info = "[%s] %s %.2f %s %s %.2f %s"
        else:
            self.info = "[%s] %s %.2f %s"
        self.title = title
        self.total = total
        self.count = count
        self.chunk_size = chunk_size
        self.status = run_status or ""
        self.fin_status = fin_status or " " * len(self.statue)
        self.unit = unit
        self.seq = sep

    def __get_info(self):
        if self.total is not None:
            _info = self.info % (self.title, self.status,
                                 self.count/self.chunk_size, self.unit, self.seq, self.total/self.chunk_size, self.unit)
        else:
            _info = self.info % (self.title, self.status, self.count / self.chunk_size, self.unit)
        return _info

    def refresh(self, count=1, status=None):
        self.count += count
        # if status is not None:
        self.status = status or self.status
        end_str = "\r"
        if self.total is not None and self.count >= self.total:
            end_str = '\n'
            self.status = status or self.fin_status
        print(self.__get_info(), end=end_str)


def load_url_google(id, name=None):
    destination_path = os.path.join(expanduser('~'), '.cache/alphavideo/checkpoint')
    os.makedirs(destination_path, exist_ok=True)
    if name is None:
        name =id
    file = os.path.join(destination_path, name)
    if not os.path.exists(file):
        print('Download checkpoint: ' + str(name))
        download_file_from_google_drive(id, destination=file)
        print('\nFinish downloading checkpoint')
    return file


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)

    token = get_confirm_token(response)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 1024
    content_size = response.headers.get('content-length')
    progress = ProgressBar(os.path.basename(destination), total=content_size,
                           unit="KB", chunk_size=CHUNK_SIZE, run_status="Downloading", fin_status="Finish downloading")

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                progress.refresh(count=len(chunk))


# load_url_google(id='1jLgyNmiZ_c-m8Cw3NcZTEPTf6VESfIzK')
