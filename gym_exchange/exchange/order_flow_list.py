class FlowList():
    '''
    A doubly linked list of Orders. Used to iterate through Orders when
    a price match is found. Each OrderList is associated with a single
    price. Since a single price match can have more quantity than a single
    Order, we may need multiple Orders to fullfill a transaction. The
    OrderList makes this easy to do. OrderList is naturally arranged by time.
    Orders at the front of the list have priority.
    '''

    def __init__(self):
        self.head_order_flow = None # first order_flow in the list
        self.tail_order_flow = None # last order_flow in the list
        self.length = 0 # number of order_flows in the list
        self.volume = 0 # sum of order_flow quantity in the list AKA share volume
        self.last = None # helper for iterating

    def __len__(self):
        return self.length

    def __iter__(self):
        self.last = self.head_order_flow
        return self

    def next(self):
        '''Get the next order_flow in the list.

        Set self.last as the next order_flow. If there is no next order_flow, stop
        iterating through list.
        '''
        if self.last == None:
            raise StopIteration
        else:
            return_value = self.last
            self.last = self.last.next_order_flow
            return return_value

    __next__ = next # python3

    def get_head_order_flow(self):
        return self.head_order_flow

    def append(self, order_flow):
        if len(self) == 0:
            order_flow.next_order_flow = None
            order_flow.prev_order_flow = None
            self.head_order_flow = order_flow
            self.tail_order_flow = order_flow
        else:
            order_flow.prev_order_flow = self.tail_order_flow
            order_flow.next_order_flow = None
            self.tail_order_flow.next_order_flow = order_flow
            self.tail_order_flow = order_flow
        self.length +=1
        self.volume += order_flow.quantity

    def remove_order_flow(self, order_flow):
        self.volume -= order_flow.quantity
        self.length -= 1
        if len(self) == 0: # if there are no more order_flows, stop/return
            return

        # Remove an order_flow from the order_flowList. First grab next / prev order_flow
        # from the order_flow we are removing. Then relink everything. Finally
        # remove the order_flow.
        next_order_flow = order_flow.next_order_flow
        prev_order_flow = order_flow.prev_order_flow
        if next_order_flow != None and prev_order_flow != None:
            next_order_flow.prev_order_flow = prev_order_flow
            prev_order_flow.next_order_flow = next_order_flow
        elif next_order_flow != None: # There is no previous order_flow
            next_order_flow.prev_order_flow = None
            self.head_order_flow = next_order_flow # The next order_flow becomes the first order_flow in the order_flowList after this order_flow is removed
        elif prev_order_flow != None: # There is no next order_flow
            prev_order_flow.next_order_flow = None
            self.tail_order_flow = prev_order_flow # The previous order_flow becomes the last order_flow in the order_flowList after this order_flow is removed

    def move_to_tail(self, order_flow):
        '''After updating the quantity of an existing order_flow, move it to the tail of the order_flowList

        Check to see that the quantity is larger than existing, update the quantities, then move to tail.
        '''
        if order_flow.prev_order_flow != None: # This order_flow is not the first order_flow in the order_flowList
            order_flow.prev_order_flow.next_order_flow = order_flow.next_order_flow # Link the previous order_flow to the next order_flow, then move the order_flow to tail
        else: # This order_flow is the first order_flow in the order_flowList
            self.head_order_flow = order_flow.next_order_flow # Make next order_flow the first

        order_flow.next_order_flow.prev_order_flow = order_flow.prev_order_flow

        # Added to resolved issue #16
        order_flow.prev_order_flow = self.tail_order_flow
        order_flow.next_order_flow = None

        # Move order_flow to the last position. Link up the previous last position order_flow.
        self.tail_order_flow.next_order_flow = order_flow
        self.tail_order_flow = order_flow

    def __str__(self):
        from six.moves import cStringIO as StringIO
        temp_file = StringIO()
        for order_flow in self:
            temp_file.write("%s\n" % str(order_flow))
        #temp_file.write("%s\n" % str(self.head_order_flow))
        return temp_file.getvalue()
    
    # def __getitem__(self, n):