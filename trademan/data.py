from shutil import ExecError
import diskcache
import os,pathlib
import pandas as pd
import yfinance as yf
import numpy as np
import datetime,pytz
import logging
import time,random
import tempfile

FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(level=20,format=FORMAT)
log = logging.getLogger('trade-data')

this_dir = pathlib.Path(__file__).parent

temp_dir = os.path.join(tempfile.gettempdir(),'trademan')

data_dir_dflt = os.path.join(temp_dir,'data')
data_dir = os.environ.get('TRADEMAN_DATA_DIR',data_dir_dflt)

db_path = os.path.join(data_dir,'assets.db')
data_db = diskcache.Cache(db_path)

media_dir_dflt = os.path.join(temp_dir,'media')
media_dir = os.environ.get('TRADEMAN_MEDIA_DIR',media_dir_dflt)

print(f'TRADEMAN DATA: {data_dir}')
print(f'TRADEMAN MEDIA: {media_dir}')
print(f'use `TRADEMAN_MEDIA_DIR` or `TRADEMAN_DATA_DIR` envvars to customize paths')
os.makedirs(data_dir,exist_ok=True)
os.makedirs(media_dir,exist_ok=True)

ticker_record = 'tickers'

dl_keys = {'perf':lambda x: x.history(period='max',interval='1d'),
            'info':lambda x: x.info}

db_set_keys = {'perf':{'expire':3600*24*5},
               'info':{'expire':3600*24*30}}

all_stocks = pd.read_csv(os.path.join(this_dir,'stock_info.csv'))
snp500 = pd.read_csv(os.path.join(this_dir,'snp500_members.csv'))
etfs = pd.read_csv(os.path.join(this_dir,'etfs.csv'))

def db_existing_symbols():
    return [s.replace('tickers/','') for s in data_db.iterkeys() if s.startswith('tickers/')]

def _dl(ticker,select=None,delay=5):
    """smartly accesses diskcache data if it exists, else download it"""
    out = {}

    failures = data_db.get(f'ticker_failures',[])
    if ticker in failures:
        log.info(f'skipping previously failed {ticker}')
        return
    
    try:
        yfobj = yf.Ticker(ticker)
        for dk,func in dl_keys.items():
            
            key = f'{dk}/{ticker.lower()}'
            if key in data_db:
                log.debug(f'db get: {ticker}')
                dlip = data_db[key]
            else:
                dlwait = delay*(1+random.random()/2)
                log.info(f'downloading {dk}/{ticker} after: {dlwait}s')
                time.sleep(dlwait)

                dlip = func(yfobj)
                set_kw = {'tag':dk,**db_set_keys.get(dk,{})}
                data_db.set(key,dlip,**set_kw)

                #add to record:
                get_time = datetime.datetime.now(tz=pytz.utc).isoformat()
                data_db.set(f'{ticker_record}/{ticker}', get_time,tag='record')

            if select is None or dk in select:
                out[dk] = dlip

        return out  
    
    except Exception as e:
        log.error(e,msg=f'issue getting: {ticker}')
        failures.append(ticker)
        data_db[f'ticker_failures'] = failures

def clear_failures():
    data_db[f'ticker_failures'] = []

def get_ticker_perf(ticker):
    dat = _dl(ticker)
    if dat is None:
        return

    dfms = pd.DataFrame(dat['perf'])
    
    dfms['perf'] = dc = (dfms['Close'] - dfms['Open'])/(dfms['Open'])
    dfms['grwth'] = np.cumprod(1+dc)-1
    dfms['highQlow'] = ec = (dfms['High'] - dfms['Low'])/(dfms['High'] + dfms['Low'])    

    return dfms

def get_tickers(tickers):
    
    if isinstance(tickers,str):
        tickers =tickers.split(',')

    log.info(f'getting {len(tickers)}| {str(list(tickers))[:100]}')

    data ={}
    for tick in tickers:
        dfms = get_ticker_perf(tick)
        if dfms is not None:
            data[tick] = dfms
    table = pd.concat(data.values(),keys=data.keys())    

    return table    

def main(clear=False):
    if clear: 
        clear_failures()
        #TODO: clear db ect via options
    get_tickers(etfs['Symbol'])
    get_tickers(snp500['Symbol'])    

if __name__ == '__main__':
    main()