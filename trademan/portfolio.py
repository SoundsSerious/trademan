import pypfopt as pfopt
import numpy as np
from trademan import data
from matplotlib.pyplot import *
import os
import datetime
import randomname
import cvxpy as cp
import argparse

def get_items(canidates,filtr:list=None):
    
    tickers = canidates
    if filtr:
        tickers = list(filter(filtr,canidates))
    print(f'filtered: {set(canidates).difference(set(tickers))}')
    df = data.get_tickers(tickers)
    return df
    

def plot_portfolio(weights,shares=None):
    """
    Plot the portfolio weights as a horizontal bar chart

    :param weights: the weights outputted by any PyPortfolioOpt optimizer
    :type weights: {ticker: weight} dict
    :param ax: ax to plot to, optional
    :type ax: matplotlib.axes
    :return: matplotlib axis
    :rtype: matplotlib.axes
    """
    fig,ax = subplots(figsize=(6,10))

    desc = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    labels = [i[0] for i in desc]
    vals = [i[1] for i in desc]

    y_pos = np.arange(len(labels))

    hbars = ax.barh(y_pos, vals)
    if shares:
        ax.bar_label(hbars,labels=[f'{v}' for v in shares.values()])
    ax.set_xlabel("Weight")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    return fig,ax

def simple_allocate(T,df,w_goal):
    dfc = df['Close'].unstack(0)
    prices = dfc[list(w_goal)].iloc[-1]
    alloc = pfopt.discrete_allocation.DiscreteAllocation(w_goal, prices,T)
    return alloc.greedy_portfolio(True)[0]

def max_allocate(T,df,w_goal):
    dfc = df['Close'].unstack(0)
    prices = {stk:dfc[stk].iloc[-1] for stk in w_goal}
    assert len(prices) == len(w_goal)    

    prices = np.array(list(prices.values()))
    w_goal = np.array(list(w_goal.values()))

    p = prices
    n = len(w_goal)
    x = cp.Variable(n,integer=True)
    r = cp.Variable()

    #minimize remainder plus norm( target - shares x price)
    objective = cp.Minimize(r+cp.norm1(T*w_goal - cp.multiply(x,p)))
    #remainder is difference of total and sum(pricesxshares)
    strict_remainder = (r+x@p == T)
    #no shorting or credit
    postive_remainder = r>=0
    constraints = [strict_remainder,postive_remainder]
    prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by `prob.solve()`.
    out = prob.solve()
    return {stk:v for stk,v in zip(wght,x.value)}

    

def make_portfolio(df,gamma=2,risk='ledoit_wolf',returns='mean',rfr=0.045/252,min_weight=0.01,max_weight=None,savefig=True,opt='sharpe',filename=None,name=None,err_factor=1,std_dev=1,allocate_amount=100000):

    pdf = df['Close'].unstack(0)

    if returns=='mean':
        mu = pfopt.expected_returns.mean_historical_return(pdf,returns_data=False)
    else:
        raise KeyError(f'bad return model: {returns}')

    if risk=='covariance':
        S = pfopt.risk_models.sample_cov(pdf,returns_data=False)
    # add downside cov semicov
    elif risk == 'ledoit_wolf':
        S = pfopt.risk_models.CovarianceShrinkage(pdf,returns_data=False).ledoit_wolf()
    else:
        raise KeyError(f'bad risk model: {risk}')

    Nl = pdf.shape[0]
    Nd = pdf.isna().sum()
    Nt = (Nl-Nd)

    s = S.to_numpy()
    si = np.sum(np.eye(s.shape[0])*s,axis=1)
    cycle_frac = (Nt/252)/10
    cycle_penalty = (1/cycle_frac)**2
    
    stderr = si*err_factor*cycle_penalty/np.sqrt(Nt)
    safelim = si*std_dev

    mu_base = mu

    #print(cycle_penalty,stderr,safelim)

    if err_factor != 0:
        mu =  np.maximum(mu - stderr,0)
    if std_dev != 0:
        mu = np.maximum(mu - safelim,0)

    # Optimize for maximal Sharpe ratio
    ef = pfopt.EfficientFrontier(mu, S)
    ef.add_objective(pfopt.objective_functions.L2_reg, gamma=gamma)

    #if min_weight:
    #    ef.add_constraint(lambda x: x>=min_weight)
    if max_weight:
        ef.add_constraint(lambda x: x<=max_weight)

    if opt == 'sharpe':
        weights = ef.max_sharpe(rfr)
    elif opt == 'min_volatility':
        weights = ef.min_volatility()
    elif opt == 'eff_return':
        weights = ef.efficient_return(0.9*mu.max()+0.1*mu.min())
    elif opt == 'eff_risk':
        weights = ef.efficient_risk(0.9*S.min()+0.1*S.max())
        

    perf = ef.portfolio_performance(True,risk_free_rate=rfr)

    warray = np.array(list(weights.values()))

    mu_act = np.sum(mu_base * warray)

    wfilt = {k:v for k,v in weights.items() if v >= min_weight and not pdf[k].isna().iloc[-1]}
    tot = np.sum(list(wfilt.values()))
    wfilt = {k:v/tot for k,v in wfilt.items()}

    shares = None
    if allocate_amount:
        shares = simple_allocate(allocate_amount,df,wfilt)

    fig,ax = plot_portfolio(wfilt,shares)
    
    if name is None:
        name = randomname.get_name(adj=('algorithms','temperature','corporate_prefixes','quantity'),noun=('accounting','corporate','algorithms'))

    if filename is None:
        filename = f'Portfolio_{name}_{opt}_{returns}_{risk}'
    
    ax.set_title(f'{name} portfolio| risk:{risk} return:{returns} opt:{opt} gamma:{gamma}\n Allocate: ${allocate_amount} Min Return: {perf[0]*100.:3.2f} Act Return: {mu_act*100:3.2f}% Ann Volatility: {perf[1]*100.:3.2f}')

    if savefig and data.media_dir:
        pth = os.path.join(data.media_dir,filename)
        fig.savefig(pth)

    return wfilt

def cli():

    parser = argparse.ArgumentParser('Portfolio Generator')
    parser.add_argument('-risk',choices=['covariance','ledoit_wolf'],default='ledoit_wolf',help='select the risk model, standard covariance or the extremity filtering `ledoit wolf` model')
    parser.add_argument('-retrn',choices=['mean'],default='mean',help='return model - mean historical performance averages')    
    parser.add_argument('-opt',choices=['sharpe','min_volatility','eff_return','eff_risk'],default='sharpe',help='optimization model: maximum risk to return model via sharpe, min_volatility only considers risk, and efficient models will try to achieve 90 percent of the best performing asset')
    parser.add_argument('-cls',choices=['etfs','stocks','all'],default='all',help='choose which type of items to trade')
    parser.add_argument('-alloc',type=int,default=None,help='choose the amount of money to allocate, this will label the output chart with the number of shares to purchase')
    parser.add_argument('-name',type=str,default=None,help='add a name to the portfolio, if none provided a randomly generated name will be created')
    parser.add_argument('-filename',type=str,default=None,help='where to store the file, by default it will be stored in a dir set by `TRADEMAN_MEDIA_DIR` ')
    parser.add_argument('-rfr',type=float,default=0.045,help='the risk free rate, adjusted per daily returns')
    parser.add_argument('-gamma',type=float,default=1,help='the weight regularizer, large values penalize small weight values, make 0 to not penalize small weights')
    parser.add_argument('-cycl-err',type=float,default=0,help='default: 0| penalize new assets returns by a factor of economic cycle: `cycle-err x standard error x (10/Nyears)^2`')
    parser.add_argument('-std-err',type=float,default=0,help='default: 0| penalize returns by subtracting the `std-err x std-dev`')
    
    parser.add_argument('-min-wght',type=float,default=0.01,help='assets less than this percent are filtered from the final portfolio')
    parser.add_argument('-max-wght',type=float,default = None,help='assets are limited to this max percentage')
    parser.add_argument('-in','--include',type=str,default=None,help='csv of a strict include on the ticker name')
    parser.add_argument('-ex','--exclude',type=str,default=None,help='csv of a strict exclude on the ticker name')

    args = parser.parse_args()

    filt = None
    inc=None
    exc=None
    if args.include or args.exclude:
        inc = args.include.split(',') if args.include else None
        exc = args.exclude.split(',') if args.exclude else None
        #print(inc,exc)
        def filt(item):
            if inc is not None and item not in inc:
                return False
            if exc is not None and item in exc:
                return False
            return True
        
    if args.cls == 'all':
        items = data.etfs['Symbol'].to_list()+data.snp500['Symbol'].to_list()
        items = list(set(data.db_existing_symbols()).union(set(items)))
    elif args.cls == 'etfs':
        items = data.etfs['Symbol'].to_list()
    elif args.cls == 'stocks':
        items = data.snp500['Symbol'].to_list()

    # add included items not in symbol list
    if inc:
        for intrd in inc:
            if intrd not in items:
                items.append(intrd)
    
    df = get_items( items,filt)

    out = make_portfolio(df,rfr=args.rfr/252,returns=args.retrn,opt=args.opt,risk=args.risk,gamma=args.gamma,allocate_amount=args.alloc,filename=args.filename,name=args.name,min_weight=args.min_wght,max_weight=args.max_wght,err_factor=args.cycl_err,std_dev=args.std_err)
    show()

    return out



if __name__ == '__main__':

    wght = cli()