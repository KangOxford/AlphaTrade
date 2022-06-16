# -*- coding: utf-8 -*-


import datetime
import random

class Singleton:
    """
    A non-thread-safe helper class to ease implementing singletons.
    This should be used as a decorator -- not a metaclass -- to the
    class that should be a singleton.
    The decorated class can define one `__init__` function that
    takes only the `self` argument. Also, the decorated class cannot be
    inherited from. Other than that, there are no restrictions that apply
    to the decorated class.
    To get the singleton instance, use the `instance` method. Trying
    to use `__call__` will result in a `TypeError` being raised.
    """

    def __init__(self, decorated):
        self._decorated = decorated

    def instance(self):
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.
        """
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)



# order应该具有流水号，交易者ID，买入或卖出， 价格， 数量 ，时间， 股票代码这些参数
class Order:

    def __init__(self, number, tid, otype, price, qty, time, stockcode):
        self.tid = tid
        self.otype = otype
        self.price = price
        self.qty = qty
        self.time = time
        self.number = number
        self.stockcode = stockcode

    def __str__(self):
        return '[%s %s %s %s P=%.2f Q=%s T=%s]' % (self.number, self.tid, self.stockcode, self.otype, self.price, self.qty, self.time)

    def decrease_qty(self, qty):
        self.qty -= qty


class Orderbook_half:
    
    def __init__(self, booktype, worstprice):
        # booktype: bids or asks?
        self.booktype = booktype
        # dictionary of orders received, indexed by number
        self.orders = {}
        # limit order book, dictionary indexed by price, with order info
        self.lob = {}
        # anonymized LOB, lists, with only price/qty info
        self.lob_anon = []
        # summary stats
        self.best_price = worstprice
        self.best_number = None
        self.worstprice = worstprice
        self.n_orders = 0  # how many orders?
        self.lob_depth = 0  # how many different prices on lob?
    
    def anonymize_lob(self):
        #生成一个排序后的只含有价格和数量的order列表
        self.lob_anon=[]
        for price in sorted(self.lob):
            self.lob_anon.append([price, self.lob[price][0]])
        
    def build_lob(self):
        #生成订单字典，键为price, 值为Order中有用的信息
        #并生成相应的anonymize_lob， 找出best price
        self.lob = {}
        for number in self.orders:
            order = self.orders[number]
            price = order.price
            if price in self.lob:
                self.lob[price][0] += order.qty
                self.lob[price][1].append([order.tid, order.qty, order.time, order.number])
            else:
                #新生成键值对，值为一个list(或者是元组)，list中有所有order的quantity的和与order list
                self.lob[price] = [order.qty, [[order.tid, order.qty, order.time, order.number]]]
        self.anonymize_lob()
        self.best_order()
    
    def book_add(self, order):
        #新增order
        self.orders[order.number] = order
        self.n_orders += 1
        self.build_lob()
        
    def book_del(self, ordernumber):
        #删除order
        if ordernumber in self.orders:
            del(self.orders[ordernumber])
            self.build_lob()
    
    def best_order(self):
        #找出最优的price和order
        self.lob_depth = len(self.lob_anon)
        if self.lob_depth > 0 :
            if self.booktype == 'bid':
                self.best_price = self.lob_anon[-1][0]
            else :
                self.best_price = self.lob_anon[0][0]
            self.best_number = self.lob[self.best_price][1][0][3]
        else :
            self.best_price = self.worstprice
            self.best_number = None
    
    def return_order(self, number):
        #查找某个特定的order
        return self.orders[number]
    
    def decrease_order_qty(self, number, qty):
        #交易中减少其中某个order的量
        self.orders[number].decrease_qty(qty)
        self.build_lob()
    
    def reset(self):
        #重置，即清空所有数据
        self.orders={}
        self.build_lob()
  
          
class Orderbook():

    def __init__(self, sys_minprice, sys_maxprice, stockcode):
        #由bid book和ask order构成，并增加一个stockcode用于限制交易的股票
        self.bids = Orderbook_half('bid', sys_minprice)
        self.asks = Orderbook_half('ask', sys_maxprice)
        self.stockcode = stockcode
        
class Exchange(Orderbook):
    #交易类
    
    def __init__(self, sys_minprice, sys_maxprice, stockcode, initprice):
        #初始化交易
        super().__init__(sys_minprice, sys_maxprice, stockcode)
        self.tape = []
        self.orderlist=[]#仅在集合竞价中用于保存所有的order,其他阶段保持空的状态
        self.price = initprice #显示当前价格，用于更新stock数据
        self.per_qty = 0#当前交易量，用于更新stock数据
        self.doneorder={}#根据成交记录生成一个order，这些order用于更新trader数据
    
    def orderlist_dec(self,order):
        #用于集合竞价中的撤单操作
        self.orderlist.remove(order)
        return self.process_order_A(datetime.datetime.now())
    
    def save_orderlist(self):
        #从连续竞价到集合竞价，保存已有的order
        for num in self.bids.orders:
            self.orderlist.append(self.bids.orders[num])
        for num in self.asks.orders:
            self.orderlist.append(self.asks.orders[num])
    
    def reset(self):
        #清空数据
        self.asks.reset()
        self.bids.reset()
        self.per_qty = 0
        self.doneorder={}
        
    def add_order(self, order):
        #向系统中增加一个order, market中不应该直接使用该函数
        if order.otype == 'bid':
            self.bids.book_add(order)
        else:
            self.asks.book_add(order)
    
    def delete_order(self, order):
        #删除order, trader撤单直接delete_order即可
        if order.otype == 'bid':
            self.bids.book_del(order.number)
        else:
            self.asks.book_del(order.number)
    
    def delete_order_by_num(self,number):
        #与上一个函数相似，为了在后续的交易过程中简化处理，特意增加该方法
        self.bids.book_del(number)
        self.asks.book_del(number)
        
            
    # this returns the LOB data "published" by the exchange,
    # i.e., what is accessible to the traders
    def publish_lob(self, time, verbose):
        public_data = {}
        public_data['time'] = time
        public_data['bids'] = {'best':self.bids.best_price,
                               'worst':self.bids.worstprice,
                               'n': self.bids.n_orders,
                               'lob':self.bids.lob_anon}
        public_data['asks'] = {'best':self.asks.best_price,
                               'worst':self.asks.worstprice,
                               'n': self.asks.n_orders,
                               'lob':self.asks.lob_anon}
        if verbose:
            print('publish_lob: t=%d' % time)
            print('BID_lob=%s' % public_data['bids']['lob'])
            print('ASK_lob=%s' % public_data['asks']['lob'])
        return public_data
    
    
    def tape_dump(self, fname, fmode, tmode):
        #输出成交记录
        dumpfile = open(fname, fmode)
        for tapeitem in self.tape:
            dumpfile.write('%s, %s\n' % (tapeitem['time'], tapeitem['price']))
        dumpfile.close()
        if tmode == 'wipe':
            self.tape = []
      
    def return_doneorder(self):
        #返回所有生成的当前已成交order 
        return self.doneorder      
        
    def save_record(self, time, price, qty, bidid, askid, bidnumber, asknumber):
        #保存每次交易的记录
        record = {'time':time,
                  'price':price,
                  'quantity':qty,
                  'bid_id':bidid,
                  'ask_id':askid,
                  'bid_number':bidnumber,
                  'ask_number':asknumber}
        self.doneorder[bidnumber] = Order(bidnumber,bidid,'bid',price,qty,time,self.stockcode)
        self.doneorder[asknumber] = Order(asknumber,askid,'ask',price,qty,time,self.stockcode)
        self.tape.append(record)
        
    def save_record_A(self,time,price,quantity,otype,tid,number):
        #集合竞价中保存交易记录，该方法目的为简化代码
        record = {'time':time,
                  'price':price,
                  'quantity':quantity,
                  'type':otype,
                  'id':tid,
                  'number':number}
        self.doneorder[number] = Order(number,tid,otype,price,quantity,time,self.stockcode)
        self.tape.append(record)
    
    def process_order_A(self, time, neworder = None, finish = False):
        #集合竞价, 竞价结束之前不断加入order，但是不成交，只输出成交价格, finish = True时该阶段结束
        #收盘集合竞价之前需要将已有的order存入orderlist中，且此时不能撤单
        if len(self.orderlist) == 0:
            self.save_orderlist()
        #先得到仅含price 和 quantity 的anonymize_lob
        self.reset()
        if neworder != None:
            #保证在该阶段结束时能通过此方法完成交易, neworder初始化为None
            self.orderlist.append(neworder)
        for order in self.orderlist:
            order_temp = Order(order.number, order.tid, order.otype, order.price, order.qty, order.time, order.number)
            self.add_order(order_temp)
        #确定使得 成交量最大 的成交价格
        #总成交量为sumqty
        sumqty = 0
        price = self.price
        while self.bids.best_price >= self.asks.best_price and(self.bids.n_orders !=0 and self.asks.best_price != 0):
            bid_order = self.bids.lob[self.bids.best_price][1][0]
            ask_order = self.asks.lob[self.asks.best_price][1][0]
            #按照价格优先和时间优先原则选出相应的买方和卖方order，交易掉其中较小的
            if bid_order[1] > ask_order[1]:
                sumqty += ask_order[1]
                price = self.asks.best_price
                self.asks.book_del(ask_order[3])
                self.bids.decrease_order_qty(bid_order[3], ask_order[1])  
            elif bid_order[1] < ask_order[1]:
                sumqty += bid_order[1]
                price = self.bids.best_price
                self.bids.book_del(bid_order[3])
                self.asks.decrease_order_qty(ask_order[3], bid_order[1])
            else:
                sumqty += bid_order[1]
                price = self.bids.best_price
                self.bids.book_del(bid_order[3])
                self.asks.book_del(ask_order[3])
        #交易完后（发生了交易才需修正），再一次修正price，使得最终所有大于price的bid均交易掉，小于price的ask均交易掉
        if sumqty != 0:
            if price < self.bids.best_price:
                price = self.bids.best_price
            if price > self.asks.best_price:
                price = self.asks.best_price
        if finish == True:
            #保存集合竞价的交易结果
            #用于处理等于price的order
            orderbid=[sumqty,[]]
            orderask=[sumqty,[]]  
            #用order列表的第一项来保存等于price的order中发生了交易的quantity
            for order in self.orderlist:
                if order.otype == 'bid' and order.price > price:
                    orderbid[0] -= order.qty
                    self.save_record_A(time,price,order.qty,'bid',order.tid,order.number)
                elif order.otype == 'ask' and order.price < price:
                    orderask[0] -= order.qty
                    self.save_record_A(time,price,order.qty,'ask',order.tid,order.number)
                elif order.price == price:
                    if order.otype == 'bid':
                        orderbid[1].append(order)
                    else:
                        orderask[1].append(order)
            #等于price的order中交易量为allqty
            for ask_order in orderask[1]:
                if orderask[0] <= 0:
                    break
                if orderask[0] <= ask_order.qty:
                    self.save_record_A(time,price,orderask[0],'ask',ask_order.tid,ask_order.number)
                    break
                self.save_record_A(time,price,ask_order.qty,'ask',ask_order.tid,ask_order.number)
                orderask[0] -= ask_order.qty
            for bid_order in orderbid[1]:
                if orderbid[0] <= 0:
                    break
                if orderbid[0] <= bid_order.qty:
                    self.save_record_A(time,price,orderbid[0],'bid',bid_order.tid,bid_order.number)
                    break
                self.save_record_A(time,price,bid_order.qty,'bid',bid_order.tid,bid_order.number)
                orderbid[0] -= bid_order.qty
            self.orderlist = []
            self.price = price
            self.per_qty = sumqty
        return price
    
    def process_order_B(self, time, order):
        #进行交易，连续竞价
        #先将order加入到orderbook中
        self.orderlist.clear()#这样可以解决连续竞价到集合竞价的衔接问题
        self.doneorder = {}
        self.add_order(order)
        sumqty = 0 #总交易量
        while self.bids.best_price >= self.asks.best_price and(self.bids.n_orders != 0 and self.asks.n_orders != 0):
            #满足需要交易的条件
            best_bid_qty=self.bids.lob[self.bids.best_price][0]
            best_ask_qty=self.asks.lob[self.asks.best_price][0]
            if order.otype == 'bid':
                #新增买方order使得达成交易条件
                for askorder in self.asks.lob[self.asks.best_price][1]:
                    present_qty = askorder[1]
                    if best_bid_qty >= present_qty:
                        #买方数量多于卖方
                        #将买方的order删除，减少quantity后重新 add， 删除当前的ask order
                        self.save_record(time, self.asks.best_price, present_qty, order.tid, askorder[0], order.number, askorder[3])               
                        sumqty += present_qty
                        self.delete_order(order)
                        if best_bid_qty > present_qty:
                            order.decrease_qty(present_qty)
                            self.add_order(order)
                        self.delete_order_by_num(askorder[3])
                        best_bid_qty -= present_qty
                    else:
                        #买方数量小于卖方
                        #删除买方order, 并相应的减少当前ask order中的quantity
                        self.save_record(time, self.asks.best_price, best_bid_qty, order.tid, askorder[0], order.number, askorder[3])
                        sumqty += best_bid_qty
                        self.asks.decrease_order_qty(askorder[3],best_bid_qty)
                        self.delete_order(order)
                        break
            else:
                #新增卖方order使得达成交易条件
                for bidorder in self.bids.lob[self.bids.best_price][1]:
                    present_qty = bidorder[1]
                    if best_ask_qty >= present_qty:
                        #卖方数量小于买方
                        #具体细节处理同上
                        self.save_record(time, self.bids.best_price, present_qty,  bidorder[0],order.tid,  bidorder[3],order.number)               
                        sumqty += present_qty
                        self.delete_order(order)
                        if best_ask_qty > present_qty:
                            order.decrease_qty(present_qty)
                            self.add_order(order)
                        self.delete_order_by_num(bidorder[3])
                        best_ask_qty -= present_qty
                    else:
                        self.save_record(time, self.bids.best_price, best_ask_qty,  bidorder[0],order.tid,  bidorder[3],order.number)
                        sumqty += best_ask_qty
                        self.bids.decrease_order_qty(bidorder[3],best_ask_qty)
                        self.delete_order(order)
                        break
        if len(self.tape) != 0:
            self.price = self.tape[-1]['price'] 
        self.per_qty = sumqty          
    
    def get_order(self):
        #返回该交易类中的所有order,返回数据不能随意修改，否则后果自负
        return_order = {}
        for ordernum in self.bids.orders:
            return_order[ordernum] = self.bids.orders[ordernum]
        for ordernum in self.asks.orders:
            return_order[ordernum] = self.asks.orders[ordernum]
        return return_order
    
    def sjprice(self,otype,quantity):
        #用于处理市价订单，根据quantity计算出最终的成交价格，作为市价单的price,同时使能撤回可能超出总量的quantity部分
        lastqty = quantity
        if otype == 'bid':
            if self.asks.lob_depth == 0:
                return (self.price, 0)
            for qtyprice in self.asks.lob_anon:
                if qtyprice[1] >= lastqty:
                    return (qtyprice[0],quantity)
                else:
                    lastqty -= qtyprice[1]
            #如果全部交易掉，则返回最终价格，并撤回多余部分
            return (self.asks.lob_anon[-1][0],quantity-lastqty)
        else:
            if self.bids.lob_depth == 0:
                return (self.price, 0)
            for qtyprice in reversed(self.bids.lob_anon):
                if qtyprice[1] >= lastqty:
                    return (qtyprice[0],quantity)
                else:
                    lastqty -= qtyprice[1]
            return (self.bids.lob_anon[0][0],quantity-lastqty)
    
    def return_info(self):
        #向外界提供需要的数据，暂时只返回self.price和quantity更新股票价格,和相关anonymize_lob信息
        return [self.price, self.per_qty, self.bids.lob_anon, self.asks.lob_anon]

class Trader:
    #交易者
    
    def  __init__(self, tid, balance, profit, stocks):
        #trader id , 余额，利润， 所拥有的order, 拥有的stock
        self.balance = balance
        self.profit = profit
        self.tid = tid
        self.orders = {}
        self.stocks = stocks #字典 stockcode为键,值为[买入均价，现价,利润，总quantity，总金额, 可委托数量]
    
    def get_stockqty(self,stockcode):
        #得到某支股票的持仓
        if stockcode in self.stocks:
            return self.stocks[stockcode][5]
        else:
            return 0
    
    def __str__(self):
        return '[TID %s balance %.2f profit %.2f orders %s stocks %s]' % (self.tid, self.balance, self.profit, self.orders, self.stocks)
               
    def delete_order(self, number, bidwithdraw = False, askwithdraw = False):
        #删除order，如果是买入撤单则应该增加余额, 如果是卖出撤单则应相应增加股票的可委托数量    
        if bidwithdraw == True:
            self.balance += self.orders[number].price * self.orders[number].qty
        if askwithdraw == True:
            stockcode = self.orders[number].stockcode
            self.stocks[stockcode][5] += self.orders[number].qty
        del(self.orders[number])
    
    def order_dec(self,number, qty):
        #成交的order只有一部分时,减少该order的quantity, 否则直接删除
        if self.orders[number].qty == qty:
            del(self.orders[number])
        else:
            self.orders[number].decrease_qty(qty)
    
    def carculate_profit(self):
        for stockcode in self.stocks:
            self.profit += self.stocks[stockcode][2]
    
    def correct_balance(self,number,price,qty):
        #买入时是先根据给定的委托价生成order，该价格可能与最终成交价不同，此时即需要修正余额
        self.balance += (self.orders[number].price - price)*qty
    
    def done_order(self, order):
        #成功交易后改变stock的内容，可以是add也可以是delete
        #order为生成的成功交易的委托, 价格为成交价，数量为成交数量，其余数据继承了原order
        if order.otype == 'bid':
            self.correct_balance(order.number,order.price,order.qty)
            if order.stockcode in self.stocks:
                stockinfo = self.stocks[order.stockcode]
                profitchange = round(stockinfo[3]*(order.price-stockinfo[1]),2)
                self.profit += profitchange
                stockinfo[2] += profitchange
                stockinfo[1] = order.price
                stockinfo[4] += order.qty*order.price + profitchange
                stockinfo[0] = round((stockinfo[0]*stockinfo[3]+order.qty*order.price)/(stockinfo[3]+order.qty),2)
                stockinfo[3] += order.qty
                stockinfo[5] += order.qty
                self.stocks[order.stockcode] = stockinfo
                self.order_dec(order.number, order.qty)
            else:
                stockinfo = [order.price, order.price, 0, order.qty, order.price*order.qty, order.qty]
                self.stocks[order.stockcode] = stockinfo
                self.order_dec(order.number, order.qty)
        else:
            #卖出
            stockinfo = self.stocks[order.stockcode]
            if stockinfo[3] <= order.qty:
                #实际上小于已在create_order限制不能取到,即此处只有等于，删除该股票及order，增加余额
                del(self.stocks[order.stockcode])
                self.delete_order(order.number)
                self.balance += order.price*order.qty
            else:
                #只卖出了一部分
                self.balance += order.price*order.qty
                stockinfo = self.stocks[order.stockcode]
                profitchange = round(stockinfo[3]*(order.price - stockinfo[1]),2)
                self.profit += profitchange
                stockinfo[2] += profitchange
                stockinfo[1] = order.price
                stockinfo[4] = stockinfo[4] - order.qty*order.price + profitchange
                stockinfo[0] = round((stockinfo[0]*stockinfo[3]-order.qty*order.price)/(stockinfo[3]-order.qty),2)
                stockinfo[3] -= order.qty
                self.stocks[order.stockcode] = stockinfo
                self.order_dec(order.number, order.qty)
                
    
    def update_stock(self,stockcode,price):
        #更新某支股票的现价，并更新self.stocks和收益，这里的price是对应stockcode的stock的市场价格。为了保证交易者的股票的价格和市场的价格是相同的。
        delta_profit=round((price-self.stocks[stockcode][1])*self.stocks[stockcode][3],2)
        #利润的增量
        self.stocks[stockcode][1]=price
        #更新对应的股票的价格
        self.stocks[stockcode][2]=self.stocks[stockcode][2]+delta_profit
        #更新对应的股票的利润
        self.stocks[stockcode][4] += delta_profit
        self.profit += delta_profit
        return
    
    def create_order(self,stockcode,otype,price,qty):
        #根据余额和stock决定是否能生成委托，更新余额，返回委托
        time=datetime.datetime.now()#委托时间
        number_time=str(time.year)+str(time.month)+str(time.day)+str(time.hour)+str(time.minute)+str(time.second)+str(time.microsecond)
        number=number_time+str(stockcode)+str(self.tid)
        order=Order(number,self.tid,otype,price,qty,time,stockcode)
        #传入Price 注意有价格限制 
        #传入qty 注意卖出时不能超过可委托数量
        self.orders[order.number] = order
        if otype=='bid':
            self.balance -= price*qty
        #改变余额账户，只考虑买股票的情况
        else:  
            self.stocks[stockcode][5] -= qty
        #改变持股账户，只考虑卖股票的情况
        return order
    
    def create_sj_order(self, stockcode, otype, qty):
        #生成市价订单，没有price，无法更新balance
        #因此，如果是买单，请注意一定要在计算出price之后立即更新balance，切记
        #对于数量应该有一个限制，即能保证当前余额大于qty*涨停价
        time = datetime.datetime.now()
        number_time=str(time.year)+str(time.month)+str(time.day)+str(time.hour)+str(time.minute)+str(time.second)+str(time.microsecond)
        number=number_time+str(stockcode)+str(self.tid)
        price = 0#先将市价订单的价格初始化为0，返回一个未完成的市价订单
        order=Order(number,self.tid,otype,price,qty,time,stockcode)
        self.orders[order.number] = order
        if otype == 'ask':
            self.stocks[stockcode][5] -= qty
        return order

class Robot1(Trader):
    #自动化交易机器人1
    #随机股票当前价格附近买入或卖出限价单
    
    def strategy(self, stocks):
        #stocks为一个字典，键为股票的code, 值为[股票的现价,涨停价，跌停价]
        chosestock = random.choice(list(stocks.keys()))
        price = stocks[chosestock][0] + random.randrange(-10,10)*0.01
        if price >= stocks[chosestock][1]:
            price = stocks[chosestock][1]
        elif price <= stocks[chosestock][2]:
            price = stocks[chosestock][2]
        otype = random.choice(['bid','ask','none'])
        if otype == 'bid':
            maxqty = self.balance//price
            quantity = random.randrange(start=0,stop=maxqty+1)
        elif otype == 'ask':
            if chosestock in self.stocks:
                maxqty = self.stocks[chosestock][5]
                quantity = random.randrange(start=0,stop=maxqty+1)
            else:
                print('可委托数量为0')
                return None
        else:
            return None
        if quantity == 0:
            return None
        order = self.create_order(chosestock,otype,price,quantity)
        return order

class Robot2(Trader):
    #自动化交易机器人2
    #随机股票买入或卖出市价单
    
    def strategy(self,stocks):
        chosestock = random.choice(list(stocks.keys()))
        price = stocks[chosestock][1]
        otype = random.choice(['bid','ask', 'none'])
        if otype == 'bid':
            maxqty = self.balance//price
            quantity = random.randrange(start=1,stop=maxqty+1)
        elif otype == 'ask':
            if chosestock in self.stocks:
                maxqty = self.stocks[chosestock][5]
                quantity = random.randrange(start=1,stop=maxqty+1)
            else:
                print('可委托数量为0')
                return None
        else:
            return None
        if quantity == 0:
            return None
        order=self.create_sj_order(chosestock,otype,quantity)
        return order

class Robot3(Trader):
    #自动化交易机器人3
    #根据以往数据决定是否买入和卖出,现价单,或者市价单
    
    def strategy(self, stockdata, qtys, sj = False):
        #stockdata包括每只股票的过往价格和现价
        #qtys应该包括所有股票计算出的买入或卖出数量
        #sj表示是否生成市价单
        #先生成一个可买入股票字典,和可卖出字典
        bidstocks = {}
        askstocks = {}
        for stockcode in stockdata:
            pre_prices = stockdata[stockcode][0]
            price = stockdata[stockcode][1]
            if sum(pre_prices)/len(pre_prices) > price:
                bidstocks[stockcode]=price
            else:
                askstocks[stockcode]=price
        #随机决定买入还是卖出
        otype = random.choice(['bid','ask'])
        if otype == 'bid':
            chosestock = random.choice(list(bidstocks.keys()))
            price = bidstocks[chosestock]
        elif otype =='ask':
            chosestock = random.choice(list(askstocks.keys()))
            if chosestock in self.stocks:
                price = askstocks[chosestock]
            else:
                print('可委托数量为0')
                return None
        if sj == True:
            order = self.create_sj_order(chosestock, otype, qtys[chosestock])
        else:
            order = self.create_order(chosestock,otype,price,qtys[chosestock])
        return order

class Stock:
    #股票
    
    def __init__(self, stockcode, price):
        self.stockcode = stockcode
        self.pre_prece = price #昨日收盘价
        self.price = price #现价
        self.volume = 0 #成交量
        self.minprice = round(price*0.9,2) #跌停价
        self.maxprice = round(price*1.1,2) #涨停价
        self.change = 0.00 #涨跌
        self.rate = 0.0000 #涨跌幅
    
    def update(self,price,quantity):
        self.price = price
        self.volume += quantity
        self.change = round(self.price - self.pre_prece, 2)
        self.rate = round(self.change/self.pre_prece, 4)


# @Singleton
class Market:
    #市场
    def __init__(self,time1,time2,time3,time4,time5,time6, stocks, traders):
        #time1为开始时间
        #time1~time2为集合竞价阶段1-1，期间可以委托，可以撤单
        #time2~time3为集合竞价阶段1-2，期间只能委托，不能撤单
        #time4~time5为连续竞价阶段
        #time5~time6为集合竞价阶段2，期间不能撤单
        #time6为结束时间
        #stocks为市场上的股票
        #traders为市场上的交易者
        self.time1 = int(time1)
        self.time2 = int(time2)
        self.time3 = int(time3)
        self.time4 = int(time4)
        self.time5 = int(time5)
        self.time6 = int(time6)
        self.stocks = stocks
        self.traders = traders
        self.exchangedic = {}
        
    def add_stock(self, stock):
        self.stocks[stock.stockcode] = stock
        self.exchangedic[stock.stockcode] = Exchange(stock.minprice, stock.maxprice, stock.stockcode, stock.price)
    
    def add_trader(self,trader):
        self.traders[trader.tid] = trader
    
    def create_exchange(self):
        self.exchangedic = {}
        for stock in self.stocks:
            self.exchangedic[stock.stockcode] = Exchange(stock.minprice, stock.maxprice, stock.stockcode, stock.price)
        
    def return_stage(self):
        #根据当前时间返回当前的阶段
        #1为集合竞价1-1,2为集合竞价1-2,3为连续竞价，4为集合竞价2, 5表示休市
        time = datetime.datetime.now()
        #提取出h,m,s转换为int
        nowtime=time.hour*10000+time.minute*100+time.second
        if self.time1 <= nowtime and nowtime <= self.time2:
            return 1
        elif self.time2 < nowtime and nowtime <=self.time3:
            return 2
        elif self.time4 <= nowtime and nowtime <=self.time5:
            return 3
        elif self.time5 < nowtime and nowtime <=self.time6:
            return 4
        else:
            return 5
    
    def withdrawal(self,order):
        #撤单
        stage = self.return_stage()
        bidwithdraw = False
        askwithdraw = False
        if order.otype == 'bid':
            bidwithdraw = True
        else:
            askwithdraw = True
        if stage == 1:
            price=self.exchangedic[order.stockcode].orderlist_dec(order)
            print(price)#显示当前集合竞价的交易价格
            self.traders[order.tid].delete_order(order.number, bidwithdraw, askwithdraw)
            print(order)#输出order，与数据库，前端等对接
        elif stage == 3:
            self.exchangedic[order.stockcode].delete_order(order)
            self.traders[order.tid].delete_order(order.number, bidwithdraw, askwithdraw)
            print(order)#输出order,与数据库，前端等对接
        elif stage == 2 or stage ==4:
            print('当前集合竞价不允许撤单')
        else:
            print('市场关闭')
        
    def add_order(self,raworder):
        #增加委托
        #为保证不影响到原来的order数据，此处重新创建一个order对象
        if raworder == None:
            return
        order = Order(raworder.number,raworder.tid,raworder.otype,raworder.price,raworder.qty,raworder.time,raworder.stockcode)
        time = datetime.datetime.now()
        stage = self.return_stage()
        chedan = False#市价单会存在部分撤单的情况
        if (stage == 1 or stage == 2 or stage == 4) and order.price != 0:
            price=self.exchangedic[order.stockcode].process_order_A(time, order)
            print(price)#显示当前集合竞价的交易价格
            print(order)#输出order,与数据库，前端等对接
        elif stage == 3:
            #连续竞价阶段，先判断此时的order是否是市价单，如果是
            #则先根据此时的委托情况确定order的价格，调用Exchange类中的方法即可
            if order.price == 0:
                sjinfo = self.exchangedic[order.stockcode].sjprice(order.otype, order.qty)
                order.price = sjinfo[0]
                raworder.price = sjinfo[0]
                #确定是否有多余的数量需要撤单
                if order.qty > sjinfo[1]:
                    chedan = True
                #如果是买单,立即修改trader余额
                if order.otype == 'bid':
                    self.traders[order.tid].balance -= order.price*order.qty
            self.exchangedic[order.stockcode].process_order_B(time, order)
            newinfo = self.exchangedic[order.stockcode].return_info()
            doneorder = self.exchangedic[order.stockcode].return_doneorder()
            for number in doneorder:
                self.traders[doneorder[number].tid].done_order(doneorder[number])
            self.stocks[order.stockcode].update(newinfo[0],newinfo[1])
            if chedan == True:
                self.withdrawal(raworder)
            print(order)#输出order,与数据库，前端等对接
        else:
            print('市场关闭或无效委托')
            bidwithdraw = False
            askwithdraw = False
            if order.otype == 'bid':
                bidwithdraw = True
            else:
                askwithdraw = True
            self.traders[order.tid].delete_order(order.number, bidwithdraw, askwithdraw)
            
            
    def save_orderlist(self):
        #用于在连续竞价过度到集合竞价时保存已有的委托数据（也可以用于一天的交易结束后
        #对未交易委托的撤单）
        for stockcode in self.exchangedic:
            self.exchangedic[stockcode].save_orderlist()
    
    def finish_A(self):
        #用于在集合竞价结束时完成所有的交易
        time = datetime.datetime.now()
        for stockcode in self.exchangedic:
            self.exchangedic[stockcode].process_order_A(time=time, finish = True)
            newinfo = self.exchangedic[stockcode].return_info()
            doneorder = self.exchangedic[stockcode].return_doneorder()
            for number in doneorder:
                self.traders[doneorder[number].tid].done_order(doneorder[number])
            self.stocks[stockcode].update(newinfo[0],newinfo[1])
        #输出开盘价
        for stockcode in self.stocks:
            price = self.stocks[stockcode].price
            print(stockcode,end=' ')
            print(price,end='\\')
    
    def update_trader_stock(self,tid):
        #根据市场上各股票的价格更新市场上某一个trader的持仓及收益
        for stockcode in self.traders[tid].stocks:
            newinfo = self.exchangedic[stockcode].return_info()
            self.traders[tid].update_stock(stockcode, newinfo[0])
        
    def show_results(self):
        #输出各股票当前的数据
        for stockcode in self.exchangedic:
            filename = stockcode+'.csv'
            self.exchangedic[stockcode].tape_dump(filename, 'w', 'wipe')

    def get_orders(self):
        #得到当前市场上所有的order,注意其中的order数据只能读取，不能修改，否则后果自负
        totle_orders = {}#order按照股票分类
        for stockcode in self.exchangedic:
            orderdics = self.exchangedic[stockcode].get_order()
            totle_orders[stockcode] = orderdics
        return totle_orders
        
    def get_input(self,tid,stockcode,otype,price,quantity):
        #得到输入，确认输入是否符合要求，如果符合则返回True，否则返回False
        #股票代码是否错误
        if stockcode not in self.stocks:
            print('股票代码错误')
            return False
        #price和quantity格式是否错误
        try:
            price_double = float(price)
        except ValueError:
            print('价格中含有非数字字符')
            return False
        try:
            qty_int = int(quantity)
        except ValueError:
            print('输入价格不是整数')
            return False
        if price_double <= 0 or qty_int <=0:
            print('价格和数量必须为正数')
            return False
        if round(price_double,2) != price_double:
            print('价格小数位数不能超过两位')
            return False
        if qty_int%100 != 0:
            print('数量必须为整百数')
            return False
        #price和quantity是否能满足市场的限制
        if price_double > self.stocks[stockcode].maxprice or price_double < self.stocks[stockcode].minprice:
            print('价格超出有效区间')
            return False
        if otype == 'bid':
            if price_double*qty_int > self.traders[tid].balance:
                print('余额不足')
                return False
        else:
            if qty_int > self.traders[tid].get_stockqty(stockcode):
                print('数量大于可委托数量')
                return False
        
        return True

class MarketInstance(Market):
    instance = None
    # singleton pattern

    def __new__(cls):
        if cls.instance is not None:
            return cls.instance
        else:
            inst = cls.instance = super(MarketInstance, cls).__new__()
            return inst

        
if __name__=='__main__':

    
    # """
    #测试集合竞价代码
    exchange = Exchange(0,100,'001', 12)
    for i in range(10):
        order1=Order(str(i),'1','bid',round(12+0.02*i,2),i+1,str(i),'001')
        order2=Order(str(i+10),'2','ask',round(12.24-0.02*i,2),i+1,str(i),'001')
        price = exchange.process_order_A('2018-5-17',order1, False)
        price = exchange.process_order_A('2018-5-17',order2,False)
        print(">>> 01 测试集合竞价代码, price:", price)
    exchange.process_order_A('2018-5-17',finish = True)
    result1=exchange.return_info()
    result2=exchange.tape  
    """
    """
    #连续竞价
    exchange = Exchange(0,100,'101', 12)
    sumqty = 0
    for i in range(10):
        order1=Order(str(i),'1','bid',round(12+0.02*i,2),i+1,str(i),'001')
        order2=Order(str(i+10),'2','ask',round(12.24-0.02*i,2),i+1,str(i),'001')
        exchange.process_order_B('2018-5-17',order1)
        sumqty += exchange.per_qty
        exchange.process_order_B('2018-5-17',order2)
        sumqty += exchange.per_qty
    result1=exchange.return_info()
    result2=exchange.tape
    print(">>> 02.01 测试连续竞价代码, result1:", result1)
    print(">>> 02.02 测试连续竞价代码, result2:", result2)
    """
    """
    #测试Stock类
    stocks = {}
    for i in range(10):
        stock = Stock('00%d'%(i), round(i * 10.12, 2))
        stocks[stock.stockcode] = stock
    stocks['004'].update(40, 1000)
    print(">>> 03 测试连续竞价代码:")
    print(stocks['004'].volume, stocks['004'].change, stocks['004'].rate)
    """
    """
    #测试Trader类
    stock = {'001':[12, 12, 0, 2000, 24000, 2000],
             '002':[21.2, 21.2, 0, 2400, 2400*21.2, 2400]}
    trader = Trader('jnx1', 100000, 0, stock)
    #print(trader)
    order = trader.create_order('001','bid',11.9,1000)
    #print(trader.orders[order.number])
    order1 = Order(order.number, order.tid, order.otype, 11.8, 500, order.time, order.stockcode) 
    #print(trader)
    trader.done_order(order1)
    #print(trader)
    #print(trader.orders[order.number])
    order = trader.create_order('003','bid',14,1000)
    #print(trader)
    trader.done_order(order)
    #print(trader)
    order = trader.create_order('002','ask',21,2400)
    order1 = Order(order.number, order.tid, order.otype, 21, 2000, order.time, order.stockcode)
    trader.done_order(order1)
    #print(trader)
    order2 = Order(order.number, order.tid, order.otype, 21, 400, order.time, order.stockcode)
    trader.done_order(order2)
    #print(trader)
    trader.update_stock('003', 14.1) 
    #print(trader)
    """
    """
    exchange = Exchange(0,100,'002', 20)
    for i in range(10):
        order1 = Order(str(i),'1','bid',round(12+0.02*i,2),i+1,str(i),'001')
        order2 = Order(str(i+10), '1', 'bid', round(12+ 0.02*i, 2), i+2, str(i),'001')
        order3 = Order(str(i+20),'2','ask',round(12.24-0.02*i,2),i+5,str(i),'001')
        exchange.process_order_A('2018-5-23',order1)
        exchange.process_order_A('2018-5-23',order2)
        price = exchange.process_order_A('2018-5-23',order3)
        print(">>> 04 测试Exchange代码:", price)
        print(price)
    exchange.process_order_A('2018-5-23',finish = True)
    result1=exchange.return_info()
    result2=exchange.tape  
    # """ 
