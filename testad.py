from Adafruit_IO import Client, Feed, RequestError


ADAFRUIT_IO_KEY = 'aio_GZYX65IsgWLAnkPa8KyajYKRVUqe'
ADAFRUIT_IO_USERNAME = 'dederohmat98'
aio = Client(ADAFRUIT_IO_USERNAME, ADAFRUIT_IO_KEY) 
feed = aio.feeds('hxt')

uhuy = "Mangga Jelek"

aio.send('hxt', uhuy)