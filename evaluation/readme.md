## Setup
urllib dependencies: sudo apt install -y gconf-service libasound2 libatk1.0-0 libc6 libcairo2 libcups2 libdbus-1-3 libexpat1 libfontconfig1 libgcc1 libgconf-2-4 libgdk-pixbuf2.0-0 libglib2.0-0 libgtk-3-0 libnspr4 libpango-1.0-0 libpangocairo-1.0-0 libstdc++6 libx11-6 libx11-xcb1 libxcb1 libxcomposite1 libxcursor1 libxdamage1 libxext6 libxfixes3 libxi6 libxrandr2 libxrender1 libxss1 libxtst6 ca-certificates fonts-liberation libappindicator1 libnss3 lsb-release xdg-utils wget
mongodb https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/
wsl: https://askubuntu.com/questions/1203689/cannot-start-mongodb-on-wsl
newspaper summaries:
  >>> import nltk
  >>> nltk.download('punkt')

## Mongodb
mongosh
use newspaper_300
db.createCollection('article')
db.article.createIndex({"domain": 1}, {unique:false, sparse:false})
db.article.createIndex({"url": 1}, {unique:true, sparse:false})