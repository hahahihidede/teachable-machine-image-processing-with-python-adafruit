from Adafruit_IO import Client, Feed, RequestError


ADAFRUIT_IO_KEY = 'YOUR AIO KEY'
ADAFRUIT_IO_USERNAME = 'YOUR USERNAME'
aio = Client(ADAFRUIT_IO_USERNAME, ADAFRUIT_IO_KEY) 
feed = aio.feeds('hxt')

uhuy = "Mangga Jelek"

aio.send('hxt', uhuy)
