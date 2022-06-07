import pyms
from csv import writer
from random import shuffle

class MSGame():    
    def __init__(self):
        self.board = pyms.Board()
        self.data = []
        self.episode_moves = []
        self.reward_list = []

    def current_state(self):
        return self.board.get_site_states()

    def available_moves(self):
        if len(self.actions())>0:return True
        return False

    def make_move(self, move, reward = False):
        pre_moves = len(self.actions())
        self.board.do_move(move)
        if reward:
            self.episode_moves.append(move)
            post_moves = len(self.actions())
            self.reward_list.append(post_moves - pre_moves)       

    def actions(self):
        actions = self.board.get_moves()
        shuffle(actions)
        return actions

    def undo_move(self, action):
        self.board.undo_move(action)
    
    def reward(self):
        return len(self.episode_moves)
    
    def get_initial(self):
        return self.board.get_initial_states()
    
    def clone():
        m = MSGame()
        m.board = self.board.copy_board()
        m.data = self.data
        m.episode_moves = self.episode_moves
        m.reward_list =  self.reward_list
        return m
    
    def write_results(self, file_name):
        row = [self.reward(), self.episode_moves]
        with open(file_name, 'a') as myfile:
            writer_object = writer(myfile)
            writer_object.writerow(row)