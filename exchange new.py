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

# Order should have parameters such as serial number, trader ID, buy or sell, price, quantity, time, and ticker symbol

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
        # Generate a sorted order list containing only price and quantity
        self.lob_anon=[]
        for price in sorted(self.lob):
            self.lob_anon.append([price, self.lob[price][0]])
        
    def build_lob(self):
        #Generates an Order dictionary with a key of price and a value of useful information in Order
        #Anonymize_lob is generated to find the best Price
        self.lob = {}
        for number in self.orders:
            order = self.orders[number]
            price = order.price
            if price in self.lob:
                self.lob[price][0] += order.qty
                self.lob[price][1].append([order.tid, order.qty, order.time, order.number])
            else:
                # Generate a new key-value pair with a list(or tuple) containing the sum of quantity of all orders and the order list
                self.lob[price] = [order.qty, [[order.tid, order.qty, order.time, order.number]]]
        self.anonymize_lob()
        self.best_order()
    
    def book_add(self, order):
        #add new order
        self.orders[order.number] = order
        self.n_orders += 1
        self.build_lob()
        
    def book_del(self, ordernumber):
        #delete order
        if ordernumber in self.orders:
            del(self.orders[ordernumber])
            self.build_lob()
    
    def best_order(self):
        #Find the best price and order
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
        #Find a particular order
        return self.orders[number]
    
    def decrease_order_qty(self, number, qty):
        # Reduce the amount of one of the orders in the transaction
        self.orders[number].decrease_qty(qty)
        self.build_lob()
    
    def reset(self):
        #Reset, that is, erase all data
        self.orders={}
        self.build_lob()
  
          
class Orderbook():

    def __init__(self, sys_minprice, sys_maxprice, stockcode):
        #Consists of a Bid book and ask Order, and adds a Stockcode to restrict trading of stocks
        self.bids = Orderbook_half('bid', sys_minprice)
        self.asks = Orderbook_half('ask', sys_maxprice)
        self.stockcode = stockcode
        
class Exchange(Orderbook):
    #trading class for the exchange
    
    def __init__(self, sys_minprice, sys_maxprice, stockcode, initprice):
        #Initial transaction
        super().__init__(sys_minprice, sys_maxprice, stockcode)
        self.tape = []
        self.orderlist=[]#Used to save all orders only in collection bids, remaining empty in other stages
        self.price = initprice # Displays current prices, used to update stock data
        self.per_qty = 0#Current trading volume, used to update stock data
        self.doneorder={}#An order is generated from the transaction record, and these orders are used to update trader data

    
    def orderlist_dec(self,order):
        #Used for collection bidding in the withdrawal order operation
        self.orderlist.remove(order)
        return self.process_order_A(datetime.datetime.now())
    
    def save_orderlist(self):
        #From continuous bidding to collective bidding, save the existing ORder
        for num in self.bids.orders:
            self.orderlist.append(self.bids.orders[num])
        for num in self.asks.orders:
            self.orderlist.append(self.asks.orders[num])
    
    def reset(self):
        #wipe data
        self.asks.reset()
        self.bids.reset()
        self.per_qty = 0
        self.doneorder={}
        
    def add_order(self, order):
        #Add an order to the system. This function should not be used directly in market
        if order.otype == 'bid':
            self.bids.book_add(order)
        else:
            self.asks.book_add(order)
    
    def delete_order(self, order):
        #Delete order from trader
        if order.otype == 'bid':
            self.bids.book_del(order.number)
        else:
            self.asks.book_del(order.number)
    
    def delete_order_by_num(self,number):
        #Similar to the previous function, this method was deliberately added to simplify processing in subsequent transactions
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
        #Output transaction record
        dumpfile = open(fname, fmode)
        for tapeitem in self.tape:
            dumpfile.write('%s, %s\n' % (tapeitem['time'], tapeitem['price']))
        dumpfile.close()
        if tmode == 'wipe':
            self.tape = []
      
    def return_doneorder(self):
        #Returns all generated current traded orders
 
        return self.doneorder      
        
    def save_record(self, time, price, qty, bidid, askid, bidnumber, asknumber):
        # Keep a record of each transaction
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
        #The collection auction saves the transaction record, this method aims to simplify the code
        record = {'time':time,
                  'price':price,
                  'quantity':quantity,
                  'type':otype,
                  'id':tid,
                  'number':number}
        self.doneorder[number] = Order(number,tid,otype,price,quantity,time,self.stockcode)
        self.tape.append(record)
    
    def process_order_A(self, time, neworder = None, finish = False):
        #Set bidding, add order continuously until the end of bidding, but no transaction, only output transaction price, finish = True when this stage ends
        # The existing order needs to be placed into the OrderList before closing the collection bidding and cannot be withdrawn at this time
        if len(self.orderlist) == 0:
            self.save_orderlist()
        # Anonymize_lob with only price and quantity is first obtained

        self.reset()
        if neworder != None:
            #At the end of the phase, the transaction is guaranteed to complete through this method, and neworder is initialized to None
            self.orderlist.append(neworder)
        for order in self.orderlist:
            order_temp = Order(order.number, order.tid, order.otype, order.price, order.qty, order.time, order.number)
            self.add_order(order_temp)
        # Determine the price at which the volume is greatest

        #sum of the trasaction sumqty
        sumqty = 0
        price = self.price
        while self.bids.best_price >= self.asks.best_price and(self.bids.n_orders !=0 and self.asks.best_price != 0):
            bid_order = self.bids.lob[self.bids.best_price][1][0]
            ask_order = self.asks.lob[self.asks.best_price][1][0]
            #Select the corresponding buyer and seller order according to the principle of price priority and time priority, and trade the smaller one
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
        #After the transaction is completed (it needs to be corrected only after the transaction occurs), price is corrected again so that all bid greater than price is traded and ask less than price is traded
        if sumqty != 0:
            if price < self.bids.best_price:
                price = self.bids.best_price
            if price > self.asks.best_price:
                price = self.asks.best_price
        if finish == True:
            # Saves the trading results of the call bid

            #Used to process orders equal to price
            orderbid=[sumqty,[]]
            orderask=[sumqty,[]]  

            # Use the first item in the order list to hold the quantity that occurred in the order equal to price
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
            #The volume in order equal to price is allqTY
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
        #To trade, to bid continuously
        #Add order to OrderBook first
        self.doneorder = {}
        self.add_order(order)
        sumqty = 0 #total amount of transactions
        while self.bids.best_price >= self.asks.best_price and(self.bids.n_orders != 0 and self.asks.n_orders != 0):
            #Meet the conditions that require a transaction
            best_bid_qty=self.bids.lob[self.bids.best_price][0]
            best_ask_qty=self.asks.lob[self.asks.best_price][0]
            if order.otype == 'bid':
                #The addition of buyer order enables the transaction condition to be reached

                for askorder in self.asks.lob[self.asks.best_price][1]:
                    present_qty = askorder[1]
                    if best_bid_qty >= present_qty:
                        # There are more buyers than sellers
                        #Delete the order of the buyer, reduce the quantity and add it again to delete the current Ask Order
                        self.save_record(time, self.asks.best_price, present_qty, order.tid, askorder[0], order.number, askorder[3])               
                        sumqty += present_qty
                        self.delete_order(order)
                        if best_bid_qty > present_qty:
                            order.decrease_qty(present_qty)
                            self.add_order(order)
                        self.delete_order_by_num(askorder[3])
                        best_bid_qty -= present_qty
                    else:
                        #The number of buyers is smaller than the number of sellers
                        # Delete the buyer order and reduce quantity in the current Ask Order accordingly

                        self.save_record(time, self.asks.best_price, best_bid_qty, order.tid, askorder[0], order.number, askorder[3])
                        sumqty += best_bid_qty
                        self.asks.decrease_order_qty(askorder[3],best_bid_qty)
                        self.delete_order(order)
                        break
            else:
                # The addition of a seller's order enables a transaction condition to be reached
                for bidorder in self.bids.lob[self.bids.best_price][1]:
                    present_qty = bidorder[1]
                    if best_ask_qty >= present_qty:
                        # The number of sellers is smaller than that of buyers
                        #Same as above for details

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
    
    def sjprice(self,otype,quantity):
        # It is used to process market order and calculate the final transaction price according to quantity as the price of market order

        if otype == 'bid':
            for qtyprice in self.asks.lob_anon:
                if qtyprice[1] >= quantity:
                    return qtyprice[0]
                else:
                    quantity -= qtyprice[1]
        else:
            for qtyprice in reversed(self.bids.lob_anon):
                if qtyprice[1] >= quantity:
                    return qtyprice[0]
                else:
                    quantity -= qtyprice[1]
    
    def return_info(self):
        # Provide the required data to the outside world, temporarily returning only self.price and quantity updates to stock prices, and anonymize_LOb information
        return [self.price, self.per_qty, self.bids.lob_anon, self.asks.lob_anon]

class Trader:
    #trader
    
    def  __init__(self, tid, balance, profit, stocks):
        #Trader ID, balance, Profit, Order owned, Stock owned
        self.balance = balance
        self.profit = profit
        self.tid = tid
        self.orders = {}
        self.stocks = stocks #The dictionary stockcode is the key and the value is [average purchase price, current price, profit, total quantity, total amount, available quantity]
    
    def __str__(self):
        return '[TID %s balance %.2f profit %.2f orders %s stocks %s]' % (self.tid, self.balance, self.profit, self.orders, self.stocks)
               
    def delete_order(self, number, bidwithdraw = False, askwithdraw = False):
        #Delete order. If it is a buy withdrawal order, the balance should be increased; if it is a sell withdrawal order, the entrusted number of stocks should be increased accordingly   
        if bidwithdraw == True:
            self.balance += self.orders[number].price * self.orders[number].qty
        if askwithdraw == True:
            stockcode = self.orders[number].stockcode
            self.stocks[stockcode][5] += self.orders[number].qty
        del(self.orders[number])
    
    def order_dec(self,number, qty):
        # If only part of the order is traded, reduce the quantity of the order; otherwise, delete the order directly
        if self.orders[number].qty == qty:
            del(self.orders[number])
        else:
            self.orders[number].decrease_qty(qty)
    
    def carculate_profit(self):
        for stockcode in self.stocks:
            self.profit += self.stocks[stockcode][2]
    
    def correct_balance(self,number,price,qty):
        #When buying, the order is first generated according to the given entrusted price, which may be different from the final transaction price, and then the balance needs to be corrected

        self.balance += (self.orders[number].price - price)*qty
    
    def done_order(self, order):
        #Change the contents of a stock after a successful trade, either by add or delete
        #Order is the commission of the generated successful transaction, price is the transaction price, quantity is the transaction quantity, and other data inherit the original order
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
            #sell
            stockinfo = self.stocks[order.stockcode]
            if stockinfo[3] <= order.qty:
                #< create_order limit (create_order limit, create_order limit)

                del(self.stocks[order.stockcode])
                self.delete_order(order.number)
                self.balance += order.price*order.qty
            else:
                #Only a fraction was sold
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
        # Update the current price of a stock and update self.stocks and earnings, where price is the market price of the stock corresponding to Stockcode. To ensure that the price of a trader's stock is the same as the market price.

        delta_profit=round((price-self.stocks[stockcode][1])*self.stocks[stockcode][3],2)
        # Incremental profit
        self.stocks[stockcode][1]=price
        #Update the corresponding stock price
        self.stocks[stockcode][2]=self.stocks[stockcode][2]+delta_profit
        # Update corresponding stock profit
        self.stocks[stockcode][4] += delta_profit
        self.profit += delta_profit
        return
    
    def create_order(self,stockcode,otype,price,qty):
        #Determines whether a delegate can be generated based on the balance and stock, updates the balance, and returns the delegate

        time=datetime.datetime.now()#time
        number_time=str(time.year)+str(time.month)+str(time.day)+str(time.hour)+str(time.minute)+str(time.second)+str(time.microsecond)
        number=number_time+str(stockcode)+str(self.tid)
        order=Order(number,self.tid,otype,price,qty,time,stockcode)
        # Pass in Price notice that there is a Price constraint
        # Pass in qTY and be careful not to sell more than the entrustable quantity
        self.orders[order.number] = order
        if otype=='bid':
            self.balance -= price*qty
        #Change the balance account and only consider stocks
        else:  
            self.stocks[stockcode][5] -= qty
        #Change the holding account to consider only selling shares

        return order
    
    def create_sj_order(self, stockcode, otype, qty):
        #Generate market order, cannot update balance without price
        #Therefore, if you are paying, be sure to update balance immediately after calculating price, remember

        #There should be a limit on the quantity, that is, to ensure that the current balance is greater than the QTY * limit
        time = datetime.datetime.now()
        number_time=str(time.year)+str(time.month)+str(time.day)+str(time.hour)+str(time.minute)+str(time.second)+str(time.microsecond)
        number=number_time+str(stockcode)+str(self.tid)
        price = -1# Initialize the market order price to -1 and return an outstanding market order
        order=Order(number,self.tid,otype,price,qty,time,stockcode)
        self.orders[order.number] = order
        if otype == 'ask':
            self.stocks[stockcode][5] -= qty
        return order

class robot1(Trader):
    # Automated trading robot 1
    #Random stock current price buy or sell limit price sheet
    
    def strategy(self, stocks):
        #Stocks is a dictionary. The key is the code of the stock. The value is [the current price of the stock, the limit up price, the limit down price].
        chosestock = random.choice(list(stocks.keys()))
        price = stocks[chosestock][0]
        otype = random.choice(['bid','ask','none'])
        if otype == 'bid':
            maxqty = self.balance//price
            quantity = random.randrange(start=0,stop=maxqty+1)
        elif otype == 'ask':
            if chosestock in self.stocks:
                maxqty = self.stocks[chosestock][5]
                quantity = random.randrange(start=0,stop=maxqty+1)
            else:
                print('quantity is 0')
                return None
        else:
            return None
        order = self.create_order(chosestock,otype,price,quantity)
        return order

class robot2(Trader):
    # Automated Trading robot ii
    # Random stock purchase or sale price list

    
    def strategy(self,stocks):
        chosestock = random.choice(list(stocks.keys()))
        price = stocks[chosestock][1]
        otype = random.choice(['bid','ask', 'none'])
        if otype == 'bid':
            maxqty = self.balance//price
            quantity = random.randrange(start=0,stop=maxqty+1)
        elif otype == 'ask':
            if chosestock in self.stocks:
                maxqty = self.stocks[chosestock][5]
                quantity = random.randrange(start=0,stop=maxqty+1)
            else:
                print('quantity is 0')
                return None
        else:
            return None
        order=self.create_sj_order(chosestock,otype,quantity)
        return order

class robot3(Trader):
    # Automated Trading robot iii
    #Decide whether to buy or sell based on past data, spot or market order
    
    def strategy(self, stockdata, qtys, sj = False):
        # Stockdata includes past and current prices for each stock
        # Qtys should include the calculated number of shares bought or sold
        # Sj indicates whether market order is generated
        # Mister becomes a buy stock dictionary, and a sell stock dictionary
        bidstocks = {}
        askstocks = {}
        for stockcode in stockdata:
            pre_prices = stockdata[stockcode][0]
            price = stockdata[stockcode][1]
            if sum(pre_prices)/len(pre_prices) > price:
                bidstocks[stockcode]=price
            else:
                askstocks[stockcode]=price
        # Decide randomly whether to buy or sell
        otype = random.choice(['bid','ask'])
        if otype == 'bid':
            chosestock = random.choice(list(bidstocks.keys()))
            price = bidstocks[chosestock]
        elif otype =='ask':
            chosestock = random.choice(list(askstocks.keys()))
            if chosestock in self.stocks:
                price = askstocks[chosestock]
            else:
                print(' The number of delegates is 0')
                return None
        if sj == True:
            order = self.create_sj_order(chosestock, otype, qtys[chosestock])
        else:
            order = self.create_order(chosestock,otype,price,qtys[chosestock])
        return order

class Stock:
    #stock class
    
    def __init__(self, stockcode, price):
        self.stockcode = stockcode
        self.pre_prece = price #Yesterday's close
        self.price = price #price
        self.volume = 0 #volume
        self.minprice = round(price*0.9,2) #upper limit
        self.maxprice = round(price*1.1,2) #lower limit
        self.change = 0.00 #change
        self.rate = 0.0000 #change rate
    
    def update(self,price,quantity):
        self.price = price
        self.volume += quantity
        self.change = round(self.price - self.pre_prece, 2)
        self.rate = round(self.change/self.pre_prece, 4)


@Singleton
class Market:
    #市场
    def __init__(self,time1,time2,time3,time4,time5,time6, stocks, traders):
        # Time1 indicates the start time

        # Time1 ~time2 is the collection bidding stage 1-1, during which can be entrusted, can withdraw orders
        #Time2 ~time3 is the 1-2 stage of collection bidding, during which only the order can be entrusted, not withdrawn
        #Time4 ~time5 is the continuous bidding stage
        # Time5 ~time6 is the collection bidding stage 2, during which orders cannot be withdrawn
        # Time6 indicates the end time
        #stocks avalible in the market
        #traders on the market
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
    
    def add_trader(self,trader):
        self.traders[trader.tid] = trader

    def get_stock(self, stockcode):
        return self.stocks[stockcode]
    
    def create_exchange(self):
        self.exchangedic = {}
        for stock in self.stocks:
            self.exchangedic[stock.stockcode] = Exchange(stock.minprice, stock.maxprice, stock.stockcode, stock.price)
        
    def return_stage(self):
        #Returns the current phase based on the current time
        # 1 for collection bidding 1-1,2 for collection bidding 1-2-3 for continuous bidding, 4 for collection bidding 2, 5 said closed
        time = datetime.datetime.now()
        # Extract h,m,s and convert to int
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
        stage = self.return_stage()
        bidwithdraw = False
        askwithdraw = False
        if order.otype == 'bid':
            bidwithdraw = True
        else:
            askwithdraw = True
        if stage == 1:
            price=self.exchangedic[order.stockcode].orderlist_dec(order)
            print(price)# Displays the trading price of the current collection bid
            self.traders[order.tid].delete_order(order.number, bidwithdraw, askwithdraw)
            print(order)# Output order, connect with database, front-end, etc
        elif stage == 3:
            self.exchangedic[order.stockcode].delete_order(order.number)
            self.traders[order.tid].delete_order(order.number, bidwithdraw, askwithdraw)
            print(order)#Output order, connect with database, front-end, etc
        elif stage == 2 or stage ==4:
            print(' Cancellation is not allowed for current collection bids')
        else:
            print('market closed')
        
    def add_order(self,order):
        # Increase the delegate
        doneorder = None
        time = datetime.datetime.now()
        stage = self.return_stage()
        if stage == 1 or stage == 2 or stage == 4:
            price=self.exchangedic[order.stockcode].process_order_A(time, order)
            print(price)# Displays the trading price of the current collection bid
            print(order)# Output order, connect with database, front-end, etc
        elif stage == 3:
            # In the continuous bidding stage, first determine whether the order at this time is the market order, if yes
            # The price of order is determined according to the delegate situation at this time, and the method in the Exchange class can be called
            if order.price == -1:
                order.price = self.exchangedic[order.stockcode].sjprice(order.otype, order.qty)
                # If you are paying, modify trader balance immediately
                if order.otype == 'bid':
                    self.traders[order.tid].balance -= order.price*order.qty
            self.exchangedic[order.stockcode].process_order_B(time, order)
            newinfo = self.exchangedic[order.stockcode].return_info()
            doneorder = self.exchangedic[order.stockcode].return_doneorder()
            for number in doneorder:
                self.traders[doneorder[number].tid].done_order(doneorder[number])
            self.stocks[order.stockcode].update(newinfo[0],newinfo[1])
            print(order)#Output order, connect with database, front-end, etc
        else:
            print('marker closed')

        if not doneorder:
            return True
        return False

    def save_orderlist(self):
        #It is used to save existing delegate data when successive bidding transitions to collective bidding

        for stockcode in self.exchangedic:
            self.exchangedic[stockcode].save_orderlist()
    
    def finish_A(self):
        # Used to complete all trades at the end of a rally bid
        time = datetime.datetime.now()
        for stockcode in self.exchangedic:
            self.exchangedic[stockcode].process_order_A(time=time, finish = True)
            newinfo = self.exchangedic[stockcode].return_info()
            doneorder = self.exchangedic[stockcode].return_doneorder()
            for number in doneorder:
                self.traders[doneorder[number].tid].done_order(doneorder[number])
            self.stocks[stockcode].update(newinfo[0],newinfo[1])
    
    def update_trader_stock(self,tid):
        #Update positions and returns of a trader in the market based on the price of each stock in the market
        for stockcode in self.traders[tid].stocks:
            newinfo = self.exchangedic[stockcode].return_info()
            self.traders[tid].update_stock(stockcode, newinfo[0])
        
        
        
        
if __name__=='__main__':
    """
    #Test the collection bidding code
    exchange = Exchange(0,100,'001', 12)
    for i in range(10):
        order1=Order(str(i),'1','bid',round(12+0.02*i,2),i+1,str(i),'001')
        order2=Order(str(i+10),'2','ask',round(12.24-0.02*i,2),i+1,str(i),'001')
        price = exchange.process_order_A('2018-5-17',order1, False)
        price = exchange.process_order_A('2018-5-17',order2,False)
        print(price)
    exchange.process_order_A('2018-5-17',finish = True)
    result1=exchange.return_info()
    result2=exchange.tape  
    """
    """
    #continuous auction
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
    """
    """
    # testing class Stock
    stocks = {}
    for i in range(10):
        stock = Stock('00%d'%(i), round(i * 10.12, 2))
        stocks[stock.stockcode] = stock
    stocks['004'].update(40, 1000)
    print(stocks['004'].volume, stocks['004'].change, stocks['004'].rate)
    """
    """
    # test ing class Trader
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
        print(price)
    exchange.process_order_A('2018-5-23',finish = True)
    result1=exchange.return_info()
    result2=exchange.tape  
    """