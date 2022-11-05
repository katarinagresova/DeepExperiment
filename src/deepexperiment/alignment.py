import numpy as np
from .utils import one_hot_encoding

class Attrament:
    
    def __init__(self):
        
        self.right = "→"
        self.down = "↓"
        self.diagonal = "↘"

    def __call__(self, gene, miRNA, score_matrix, opening_percentile = 99, elonging_percentile = 90):
        
        # compute alignment for one pair of miRNA and gene
        assert type(miRNA) == str 
        assert type(gene) == str
        #assert score_matrix.shape == (len(gene), len(miRNA))

        score_matrix, opening_penalty, elonging_penalty = self._preprocess_score(miRNA, gene, score_matrix, opening_percentile, elonging_percentile)
        grid = self._forward_pass(gene, miRNA, score_matrix, opening_penalty, elonging_penalty)
        return self._backward_pass(gene, miRNA, grid, score_matrix)
    
    def _score(self, grid, i, j, d, score):
        dirs = [self.right, self.diagonal, self.down]
        s = [ grid[i, j - 1][1] - d, 
            grid[i - 1, j - 1][1] + score,
            grid[i - 1, j][1] - d ]
        return (dirs[s.index(max(s))], max(s))

    def _forward_pass(self, x, y, scor_mat, opening, elonging):
        nucl_x = [""] + [ch for ch in x]
        nucl_y = [""] + [ch for ch in y]
        
        grid = np.empty(shape = (len(nucl_x), len(nucl_y)), dtype = object)
        for i in range(0, len(nucl_x)):
            grid[i, 0] = (self.down, 0)
        for i in range(0, len(nucl_y)):
            grid[0, i] = (self.right, 0)
        
        is_opening = False
        for i in range(1, len(nucl_x)):
            for j in range(1, len(nucl_y)):
                if is_opening:
                    gap = opening
                else:
                    gap = elonging
                if i == len(nucl_x) -1 or j == len(nucl_y) - 1:
                    gap = 0
                grid[i, j] = self._score(grid, i, j, gap, scor_mat[i, j])
                if grid[i, j][0] == self.diagonal:
                    is_opening = True
                else:
                    is_opening = False

        return grid

    def _backward_pass(self, x, y, grid, score_matrix):

        nucl_x = [""] + [ch for ch in x]
        nucl_y = [""] + [ch for ch in y]

        aligned_x = []
        aligned_y = [] 
        aligned_s = []
        i = grid.shape[0] - 1
        j = grid.shape[1] - 1
        while ( i !=0 or j != 0):
            if grid[i, j][0] == self.right:
                aligned_x.append("-")
                aligned_y.append(nucl_y[j])
                aligned_s.append(0)
                j -= 1
            elif grid[i, j][0] == self.down:
                aligned_x.append(nucl_x[i])
                aligned_y.append("-")
                aligned_s.append(0)
                i -= 1
            elif grid[i, j][0] == self.diagonal:
                aligned_x.append(nucl_x[i])
                aligned_y.append(nucl_y[j])
                aligned_s.append(score_matrix[i, j])
                i -= 1
                j -= 1
        
        return (aligned_x, aligned_s, aligned_y)

    def _preprocess_score(self, miRNA, gene, score_matrix, opening_percentile, elonging_percentile):

        data = one_hot_encoding(miRNA, gene)[0,:,:,0]

        # flip the matrix to have it in the correct orientation
        score_matrix = np.flip(score_matrix, axis=1)

        # remove the negative attribution scores
        score_matrix[score_matrix < 0] = 0
        # swap sign of scores for the mismatch positions
        data[data == 0] = -1
        score_matrix = data * score_matrix
        # scale the scores to the <-1,1> interval and keep the zero
        max_absolute_value = max(abs(score_matrix.flatten()))
        score_matrix = score_matrix / max_absolute_value

        # set the opening and elonging penalties
        abs_vals = np.stack(np.abs(score_matrix.sum(-1)), 0).flatten()
        opening_gap = np.nanpercentile(abs_vals, opening_percentile)
        elonging_gap = np.nanpercentile(abs_vals, elonging_percentile)

        

        # add the first row and column of zeros - needed for computing alignment
        newrow = np.zeros(score_matrix.shape[1])
        score_matrix = np.vstack([newrow, score_matrix])
        newcol = np.zeros((score_matrix.shape[0], 1))
        score_matrix = np.hstack([newcol, score_matrix])

        return score_matrix, opening_gap, elonging_gap