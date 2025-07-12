import json
import numpy as np
import pickle
from tqdm import tqdm
from numba import jit, prange


@jit(nopython=True, parallel=True)
def _update_adjacency_matrix_1d(adjacency_matrix, images):
    """Update adjacency matrix for 1D image sequences."""
    for image in prange(len(images)):
        for i in range(len(images[image]) - 1):
            adjacency_matrix[images[image][i], images[image][i+1]] += 1
    return adjacency_matrix


@jit(nopython=True, parallel=True)
def _update_adjacency_matrix_2d(adjacency_matrix, images):
    """Update adjacency matrix for 2D images with horizontal, vertical, and diagonal patterns."""
    for image in prange(len(images)):
        height, width = images[image].shape
        for i in range(height):
            for j in range(width):
                if j < width - 1:  # horizontal
                    adjacency_matrix[images[image][i, j], images[image][i, j+1]] += 1
                if i < height - 1:  # vertical
                    adjacency_matrix[images[image][i, j], images[image][i+1, j]] += 1
                if i < height - 1 and j < width - 1:  # diagonal
                    adjacency_matrix[images[image][i, j], images[image][i+1, j+1]] += 1
    return adjacency_matrix


@jit(nopython=True)
def _apply_merge_1d(image, token1, token2, new_token):
    """Apply BPE merge for 1D sequences."""
    new_image = np.zeros(len(image), dtype=np.int32)
    idx = 0
    i = 0
    while i < len(image):
        if i < len(image) - 1 and image[i] == token1 and image[i+1] == token2:
            new_image[idx] = new_token
            i += 2
        else:
            new_image[idx] = image[i]
            i += 1
        idx += 1
    return new_image[:idx]


@jit(nopython=True)
def _apply_merge_2d(image, token1, token2, new_token):
    """Apply BPE merge for 2D images with horizontal, vertical, and L-shape patterns."""
    height, width = image.shape
    new_image = image.ravel()
    mask = np.zeros(height * width, dtype=np.bool_)

    for i in range(height):
        for j in range(width):
            idx = i * width + j
            if new_image[idx] == token1:
                if j < width - 1 and new_image[idx + 1] == token2 and not mask[idx] and not mask[idx + 1]:
                    new_image[idx] = new_token
                    mask[idx + 1] = True
                elif i < height - 1 and new_image[idx + width] == token2 and not mask[idx] and not mask[idx + width]:
                    new_image[idx] = new_token
                    mask[idx + width] = True
                elif (i < height - 1 and j < width - 1 and 
                      new_image[idx + 1] == token2 and new_image[idx + width] == token2 and
                      not mask[idx] and not mask[idx + 1] and not mask[idx + width]):
                    new_image[idx] = new_token
                    mask[idx + 1] = True
                    mask[idx + width] = True

    return new_image[~mask]


class vBPE:
    """Visual Byte-Pair Encoding tokenizer for image tokens."""
    
    def __init__(self, init_size, vocab_pad, vocab_end):
        self.vocab_pad = vocab_pad
        self.vocab_end = vocab_end
        self.vocab_start = init_size + vocab_pad
        self.encode_cache = {}
        self.total_size = init_size
        self.adjacency_matrix = np.zeros((init_size, init_size), dtype=np.int32)

    def load_vocab(self, vbpe_path):
        """Load trained vBPE vocabulary from file."""
        with open(vbpe_path, 'rb') as f:
            self.vbpe_vocab = pickle.load(f)
        print(f"vBPE successfully loaded: {vbpe_path}")

    def _update_adjacency_matrix(self, images):
        """Update adjacency matrix based on token co-occurrence patterns."""
        self.adjacency_matrix = np.zeros((self.total_size, self.total_size), dtype=np.int32)
        if images[0].ndim == 1:
            self.adjacency_matrix = _update_adjacency_matrix_1d(self.adjacency_matrix, images)
        else:
            self.adjacency_matrix = _update_adjacency_matrix_2d(self.adjacency_matrix, images)

    def _get_most_frequent_pair(self):
        """Find the most frequent token pair in the adjacency matrix."""
        upper_tri = np.triu(self.adjacency_matrix, k=1)
        if np.sum(upper_tri) == 0:
            return None, 0
        i, j = np.unravel_index(np.argmax(upper_tri), upper_tri.shape)
        return (int(i), int(j)), upper_tri[i, j]

    def _apply_merge(self, image, pair, new_token):
        """Apply BPE merge operation to an image."""
        token1, token2 = pair
        if image.ndim == 1:
            return _apply_merge_1d(image, token1, token2, new_token)
        else:
            return _apply_merge_2d(image, token1, token2, new_token)

    def train(self, train_codes, num_merges):
        """Train vBPE tokenizer on image token data."""
        self.vbpe_vocab = []
        images = train_codes
        with tqdm(total=num_merges, desc="Training vBPE") as pbar:
            for i in range(num_merges):
                self._update_adjacency_matrix(images)
                best_pair, freq = self._get_most_frequent_pair()
                if freq == 0:
                    break
                new_token = self.total_size
                self.total_size += 1
                self.vbpe_vocab.append((best_pair, new_token))
                images = [self._apply_merge(img, best_pair, new_token) for img in images]
                pbar.update(1)

    def save_vocab(self, save_path):
        """Save trained vBPE vocabulary to file."""
        with open(save_path, 'wb') as f:
            pickle.dump(self.vbpe_vocab, f)
        print(f"vBPE successfully saved: {save_path}")

    def merge_token(self, image_token):
        """Apply all learned BPE merges to compress image tokens."""
        for (token1, token2), new_token in self.vbpe_vocab:
            image_token = self._apply_merge(image_token, (token1, token2), new_token)
        return image_token

    def trans_token(self, image_token):
        """Transform image tokens with vocabulary shifting and vBPE compression."""
        image_token = np.array(image_token)
        image_token_2d = image_token.reshape(32, 32) - self.vocab_pad
        image_token_trans = self.merge_token(image_token_2d) + self.vocab_pad
        image_token_trans[image_token_trans >= self.vocab_start] += (self.vocab_end - self.vocab_start)
        return image_token_trans

    def add_trans_token(self, dt):
        """Add transformed tokens to data item with caching."""
        image_token = dt['token_image']
        if dt['image'] in self.encode_cache:
            dt['token_image_new'] = self.encode_cache[dt['image']]
        else:
            image_token_trans = self.trans_token(image_token)
            dt['token_image_new'] = image_token_trans.tolist()
            self.encode_cache[dt['image']] = dt['token_image_new']
        return dt

    def parallel_add_trans_token(self, tokenize_result):
        """Process multiple data items with progress tracking."""
        with tqdm(total=len(tokenize_result), desc="Adding trans tokens") as pbar:
            results = []
            for dt in tokenize_result:
                results.append(self.add_trans_token(dt))
                pbar.update(1)
        self.tokenize_result = results

    def save_tokenize_result(self, save_path):
        """Save tokenized results to JSONL file."""
        with open(save_path, 'w') as f:
            for item in self.tokenize_result:
                f.write(json.dumps(item) + '\n')