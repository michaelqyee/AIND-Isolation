"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # if a move causes the player to lose, the score should be the worst it can be
    if game.is_loser(player):
        return float("-inf")

    # if a move causes the player to win, the score should be the worst it can be
    if game.is_winner(player):
        return float("inf")

    # define lists of moves for each player
    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))

    own_moves_score = 0
    opp_moves_score = 0

    # iterate over the movelist for the active player
    for move in own_moves:
        # sum up all of the possible moves resulting from executing the current move
        own_moves_score += len(game._Board__get_moves(move))

    # iterate over the movelist for the active player
    for move in opp_moves:
        # sum up all of the possible moves resulting from executing the current move
        opp_moves_score += len(game._Board__get_moves(move))

    # return the difference between the possible moves unlocked by the current move
    return float(own_moves_score - opp_moves_score)


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # if a move causes the player to lose, the score should be the worst it can be
    if game.is_loser(player):
        return float("-inf")

    # if a move causes the player to win, the score should be the worst it can be
    if game.is_winner(player):
        return float("inf")

    # defining the beginning of the game as 25% blank spaces, check if we've passed the beginning of the game
    if (len(game.get_blank_spaces()) < 0.75 * game.height * game.width):
        # define the set of moves for each player
        own_moves_set = set(game.get_legal_moves(player))
        opp_moves_set = set(game.get_legal_moves(game.get_opponent(player)))
        # favor moves that offer the ability to block the opponent by selecting moves that lead to move overlap with the opponent
        return float(len(own_moves_set.intersection(opp_moves_set)))

    # otherwise, if still at the beginning of the game, count moves for both players
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    # return difference between number of moves available, weighting more importance on the active player's moves to create a solid foundation
    return float(3 * own_moves - opp_moves)


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # if a move causes the player to lose, the score should be the worst it can be
    if game.is_loser(player):
        return float("-inf")

    # if a move causes the player to win, the score should be the worst it can be
    if game.is_winner(player):
        return float("inf")


    # count moves for both players
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    # return difference between number of moves available, weighting more importance on the active player's moves to create a solid foundation
    return float(3 * own_moves - opp_moves)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def max_value(self, game, depth):
        # check whether the timer has expired
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # define recursive end condition as whether there are no moves left to execute or we've hit the target depth
        if not bool(game.get_legal_moves()) or depth == 0: 
            # if recursive end condition met, return the evaluation function of the current board state for the player
            return self.score(game, self)

        # initialize the max to negative infinity so that it will get set to any other value
        max_val = float("-inf")
        # call the min function for the next level of the tree for each move, decreasing the depth by one to signify going down the tree
        for move in game.get_legal_moves():
            # retain the maximum value across all nodes analyzed
            max_val = max(max_val, self.min_value(game.forecast_move(move), depth - 1))
        return max_val

    def min_value(self, game, depth):
        # check whether the timer has expired
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        # define recursive end condition as whether there are no moves left to execute or we've hit the target depth
        if not bool(game.get_legal_moves()) or depth == 0:
            # if recursive end condition met, return the evaluation function of the current board state for the player
            return self.score(game, self)
        
        # initialize the min to positive infinity so that it will get set to any other value
        min_val = float("inf")
        # call the max function for the next level of the tree for each move, decreasing the depth by one to signify going down the tree
        for move in game.get_legal_moves():
            # retain the minimum value across all nodes analyzed
            min_val = min(min_val, self.max_value(game.forecast_move(move), depth - 1))
        return min_val

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        # check whether the timer has expired
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # initialize the value to negative infinity so that it will get set to any other value (since the starting player is max)
        best_val = float("-inf")
        legal_moves = game.get_legal_moves()

        # check to see if there are no moves available and return according to API
        if not legal_moves:
            return (-1, -1)
        
        # initialize a best move to one of the legal moves randomly in case the first ply evaluation times out before completing
        best_move = legal_moves[random.randint(0, len(legal_moves) - 1)]

        for move in legal_moves:
            # kick off the minimax recursion and retain the maximum value and associated best move
            (best_val, best_move) = max((best_val, best_move), (self.min_value(game.forecast_move(move), depth - 1), move))

        return best_move


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        legal_moves = game.get_legal_moves()

        # check to see if there are no moves available and return according to API
        if not legal_moves:
            return (-1, -1)
        # initialize a best move to one of the legal moves randomly in case the first ply evaluation times out before completing
        best_move = legal_moves[random.randint(0, len(legal_moves) - 1)]

        try:
            iterative_depth = 1
        # The try/except block will automatically catch the exception
        # raised when the timer is about to expire.

        # evaluate increasing depths until the timer runs out
            while True:
                best_move = self.alphabeta(game, iterative_depth)
                iterative_depth += 1
        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def max_value(self, game, depth, alpha, beta):
        # check whether the timer has expired
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # define recursive end condition as whether there are no moves left to execute or we've hit the target depth
        if not bool(game.get_legal_moves()) or depth == 0: 
            # if recursive end condition met, return the evaluation function of the current board state for the player
            return self.score(game, self)
        
        # initialize the max to negative infinity so that it will get set to any other value
        max_val = float("-inf")
        # call the min function for the next level of the tree for each move, decreasing the depth by one to signify going down the tree
        for move in game.get_legal_moves():
            max_val = max(max_val, self.min_value(game.forecast_move(move), depth - 1, alpha, beta))
            # check to see if further evaluation is skippable by alpha-beta pruning and return if so
            if (max_val >= beta):
                return max_val
            # set alpha for future alpha-beta pruning in the min_value function
            alpha = max(alpha, max_val)
        return max_val

    def min_value(self, game, depth, alpha, beta):
        # check whether the timer has expired
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        # define recursive end condition as whether there are no moves left to execute or we've hit the target depth
        if not bool(game.get_legal_moves()) or depth == 0:
            # if recursive end condition met, return the evaluation function of the current board state for the player
            return self.score(game, self)
        
        # initialize the min to positive infinity so that it will get set to any other value
        min_val = float("inf")
        # call the min function for the next level of the tree for each move, decreasing the depth by one to signify going down the tree
        for move in game.get_legal_moves():
            min_val = min(min_val, self.max_value(game.forecast_move(move), depth - 1, alpha, beta))
            # check to see if further evaluation is skippable by alpha-beta pruning and return if so
            if (min_val <= alpha):
                return min_val
            # set beta for future alpha-beta pruning in the max_value function
            beta = min(beta, min_val)
        return min_val

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        # check whether the timer has expired
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # initialize the value to negative infinity so that it will get set to any other value (since the starting player is max)
        best_val = float("-inf")
        legal_moves = game.get_legal_moves()
        # check to see if there are no moves available and return according to API
        if not legal_moves:
            return (-1, -1)
        
        # initialize a best move to one of the legal moves randomly in case the first ply evaluation times out before completing
        best_move = legal_moves[random.randint(0, len(legal_moves) - 1)]

        for move in legal_moves:
            # kick off the minimax with alpha-beta pruning recursion and retain the value
            val = max(self.min_value(game.forecast_move(move), depth - 1, alpha, beta), best_val)
            # check to see if the value is a new max
            if (best_val < val):
                # save the new highest value and associated move
                best_val = val
                best_move = move
                # check to see if alpha-beta pruning can skip the rest of the evaluations and return the move if so
                if (val >= beta):
                    return best_move
            # save alpha for future alpha-beta pruning for the min_value function
            alpha = max(alpha, val)

        return best_move