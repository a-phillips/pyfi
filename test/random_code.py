#------------------------------------------------------------------------------------
#Probably not using this code
#------------------------------------------------------------------------------------

#Asian options typically aren't priced using binomial trees

class AsianFixed(object):


    def __init__(self, S, K, T, n, r, q=0, call=True, geom=False):
        _check_option_error(S, K, T, n, r, q=q, call=call, geom=geom)
        self.S = S
        self.K = K
        self.T = T
        self.n = n
        self.dt = float(T)/n
        self.r = r
        self.q = q
        self.call = call
        self.avg = ('Geometric'*geom)+('Arithmetic'*(not geom))
        self.price_tree = []
        self.stock_tree = self.make_tree()
        self.price = self.calc_price()

    def make_tree(self):
        stock_tree = []
        u = math.exp(self.S.sigma*(self.dt**.5))
        d = 1/u
        for t in xrange(self.n+1):
            prices = []
            for num_u in xrange(t+1):
                prices.append(round(self.S.S0*(u**num_u)*(d**(t-num_u)),2))
            stock_tree.append(prices)
        return stock_tree

    def get_paths(self):
        paths = []
        curr_path = []
        paths = self._iter_thru_tree(self.stock_tree, paths, curr_path, 0, 0)
        return paths

    def _iter_thru_tree(self, tree, paths, curr_path, t, i):
        if t == len(tree)-1:
            paths.append(curr_path + [tree[t][i+1]])
            paths.append(curr_path + [tree[t][i]])
            return paths
        else:
            curr_path += [tree[t][i]]
            paths = self._iter_thru_tree(tree, paths, curr_path, t+1, i)
            if t > 0:
                curr_path = curr_path[:t] + [tree[t][i+1]]
                paths = self._iter_thru_tree(tree, paths, curr_path, t+1, i+1)
        return paths

    def calc_price(self):
        self.stock_tree = self.make_tree()
        price_tree = copy.deepcopy(self.stock_tree)
        u = math.exp(self.S.sigma*(self.dt**.5))
        d = 1/u
        #pu and pd represent risk-neutral probabilities of up and down movement, respectively
        pu = (math.exp((self.r - self.q)*self.dt) - d)/(u - d)
        pd = 1 - pu
        #All paths arriving at the same end result occur with the same probability, so the
        #averages are arrived at by finding the average for each path according to the
        #specified method, adding all averages for each end price, then dividing that sum
        #by the total number of ways that path occurs. I realize that those counts will be
        #nCk, but I figured it was easier to just keep count while iterating through
        #the paths than calculating factorials.
        paths = self.get_paths()
        avgs = [0]*(self.n+1)
        counts = [0]*(self.n+1)
        for i, path in enumerate(paths):
            if self.avg == 'Arithmetic':
                final_avg = sum(path)/(self.n + 1)
            else:
                final_avg = 1
                for price in path:
                    final_avg *= price
                final_avg = (final_avg**(1.0/(self.n+1)))
            index = self.stock_tree[-1].index(path[-1])
            avgs[index] += final_avg
            counts[index] += 1
        #Find the final period payouts as difference between avg stock price and strike
        for i, num in enumerate(avgs):
            avgs[i] = float(num)/counts[i]
            if self.call:
                price_tree[-1][i] = max(avgs[i] - self.K, 0)
            else:
                price_tree[-1][i] = max(self.K - avgs[i], 0)
        #Find price at t = 0 by starting at second-to-last time period, finding the value at
        #each node in that period as the pv of the risk-neutral expected value, then repeating
        #for prior time periods
        for t in xrange(self.n-1, -1, -1):
            for i in xrange(t+1):
                price_tree[t][i] = (pd*price_tree[t+1][i])+(pu*price_tree[t+1][i+1])
                price_tree[t][i] = round(price_tree[t][i]*math.exp((self.r-self.q)*self.dt*-1), 2) #discount
        self.price = price_tree[0][0]
        self.price_tree = price_tree
        return self.price

    def print_tree(self, stock=False):
        if stock:
            for line in self.stock_tree:
                print '\t'.join([str(price) for price in line])
        else:
            for line in self.price_tree:
                print '\t'.join([str(price) for price in line])


class AsianFloating(object):


    def __init__(self, S, T, n, r, q=0, call=True, geom=False):
        _check_option_error(S, K, T, n, r, q=q, call=call, geom=geom)
        self.S = S
        self.T = T
        self.n = n
        self.dt = float(T)/n
        self.r = r
        self.q = q
        self.call = call
        self.avg = ('Geometric'*geom)+('Arithmetic'*(not geom))
        self.price_tree = []
        self.stock_tree = self.make_tree()
        self.price = self.calc_price()

    def make_tree(self):
        stock_tree = []
        u = math.exp(self.S.sigma*(self.dt**.5))
        d = 1/u
        for t in xrange(self.n+1):
            prices = []
            for num_u in xrange(t+1):
                prices.append(round(self.S.S0*(u**num_u)*(d**(t-num_u)),2))
            stock_tree.append(prices)
        return stock_tree

    def get_paths(self):
        paths = []
        curr_path = []
        paths = self._iter_thru_tree(self.stock_tree, paths, curr_path, 0, 0)
        return paths

    def _iter_thru_tree(self, tree, paths, curr_path, t, i):
        if t == len(tree)-1:
            paths.append(curr_path + [tree[t][i+1]])
            paths.append(curr_path + [tree[t][i]])
            return paths
        else:
            curr_path += [tree[t][i]]
            paths = self._iter_thru_tree(tree, paths, curr_path, t+1, i)
            if t > 0:
                curr_path = curr_path[:t] + [tree[t][i+1]]
                paths = self._iter_thru_tree(tree, paths, curr_path, t+1, i+1)
        return paths

    def calc_price(self):
        self.stock_tree = self.make_tree()
        price_tree = copy.deepcopy(self.stock_tree)
        u = math.exp(self.S.sigma*(self.dt**.5))
        d = 1/u
        #pu and pd represent risk-neutral probabilities of up and down movement, respectively
        pu = (math.exp((self.r - self.q)*self.dt) - d)/(u - d)
        pd = 1 - pu
        #All paths arriving at the same end result occur with the same probability, so the
        #averages are arrived at by finding the average for each path according to the
        #specified method, adding all averages for each end price, then dividing that sum
        #by the total number of ways that path occurs. I realize that those counts will be
        #nCk, but I figured it was easier to just keep count while iterating through
        #the paths than calculating factorials.
        paths = self.get_paths()
        avgs = [0]*(self.n+1)
        counts = [0]*(self.n+1)
        for i, path in enumerate(paths):
            if self.avg == 'Arithmetic':
                final_avg = sum(path)/(self.n + 1)
            else:
                final_avg = 1
                for price in path:
                    final_avg *= price
                final_avg = (final_avg**(1.0/(self.n+1)))
            index = self.stock_tree[-1].index(path[-1])
            avgs[index] += final_avg
            counts[index] += 1
        #Find the final period payouts as difference between final stock price and
        #calculated strike price
        for i, num in enumerate(avgs):
            avgs[i] = float(num)/counts[i]
            if self.call:
                price_tree[-1][i] = max(price_tree[-1][i] - avgs[i], 0)
            else:
                price_tree[-1][i] = max(avgs[i] - price_tree[-1][i], 0)
        #Find price at t = 0 by starting at second-to-last time period, finding the value at
        #each node in that period as the pv of the risk-neutral expected value, then repeating
        #for prior time periods
        for t in xrange(self.n-1, -1, -1):
            for i in xrange(t+1):
                price_tree[t][i] = (pd*price_tree[t+1][i])+(pu*price_tree[t+1][i+1])
                price_tree[t][i] = round(price_tree[t][i]*math.exp((self.r-self.q)*self.dt*-1), 2) #discount
        self.price = price_tree[0][0]
        self.price_tree = price_tree
        return self.price

    def print_tree(self, stock=False):
        if stock:
            for line in self.stock_tree:
                print '\t'.join([str(price) for price in line])
        else:
            for line in self.price_tree:
                print '\t'.join([str(price) for price in line])


