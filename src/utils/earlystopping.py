


class EarlyStopping():
    def __init__(self, patience , delta ):
        '''
        patience : How long to wait after last time validation - improved.
        delta : Minimum change in the monitored quantity to qualify as an improvement.

        '''
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter =0
        self.should_stop = False 


    def step(self,score):

        # If this is the first score, store it
        if self.best_score is None:
            self.best_score = score

        # Check if there's enough improvement
        elif score > self.best_score + self.delta:
            self.best_score = score
            self.counter = 0       # Reset the counter
        
        else:
            self.counter += 1 
            if self.counter >= self.patience:
                self.should_stop = True

