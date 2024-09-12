import numpy as np
import pywrapfst as fst
from itertools import product
from TL_funcs import fst2word_list
from TL_funcs import fst2tables, auto_key2fst_key
from scipy.sparse.linalg import svds
from scipy.linalg import svd

# Function to pair two natural numbers into a single natural number with cantor bijection
def cantor_pairing(a, b):
    """Maps pair of natural numbers to a single natural number with Cantor bijection

    Args:
        a (numpy natural number): 1st entry in natural ordered pair
        b (numpy natural number): 2nd entry in natural ordered pair

    Returns:
        c: natural number cantor image of the input pair
    """
    c = (a + b) * (a + b + 1) // 2 + b
    return c

# Function to retrieve the original pair from the cantor natural number
def inverse_cantor_pairing(z):
    """Maps natural numbers to an ordered pair of natural numbers with the inverse cantor bijection

    Args:
        z (numpy natural number): 1D natural number input to inverse cantor unfction

    Returns:
        x (numpy natural number): 1st entry in Cantor natural ordered pair associated with input z
        y (numpy natural number): 2nd entry in Cantor natural ordered pair associated with input z
    """ 
    w = int((8 * z + 1) ** 0.5 - 1) // 2
    t = (w * w + w) // 2
    y = z - t
    x = w - y
    return x, y

def P2Basis(P):
    """Creates new change of basis matrix based on a Hankel factorization P

    Args:
        P (|P|xn numpy array): 1st factor in Hankel basis factorization H = PS. 
        P describes the prefix of words in set P

    Returns:
        B (nxn numpy array): invertiable matrix that maps transition matrices associated with H to their deterministic representation.
    """
    n = np.linalg.matrix_rank(P)
    Binc = P[0, :]
    counter = 1

    while np.linalg.matrix_rank(Binc) < n:
        V_add = P[counter, :]
        Baug = np.vstack((Binc, V_add))

        if np.linalg.matrix_rank(Binc) < np.linalg.matrix_rank(Baug):
            Binc = Baug

        counter += 1

    B = Binc
    return B

# Spectral Decommposition algorithm
def WFA_SPEC_DECOMP(H_z):
    """Function that computes learned WFA from sample hankel matrices using SVD reconstruction algorithm.

    INPUTS
    ------
    H_z : |P|x|S|xm+1 numpy array 
        H_z = [H_e| H_x1| ... | H_xm] where H_x(a,b) = H_b(ax,b), denotes concatenation into page and basis = (P',S)
    
    OUTPUTS
    -------
    t_0 : 1xn row vector numpy array
        initial weights of learned WFA
    T_array :  nxnxm numpy array 
        concatenation of transition matrices with size form [ A_x1| A_x2| ...| A_xm ]
    t_inf : nx1 column vector numpy array
        Final weights of learned WFA    
    """
    
    Psize, Ssize, num_subblocks = H_z.shape 
    #num_subblocks = #letters + 1 because H_z includes a subblock 
    #for each letter and the original basis hankel subblock
    num_letters = num_subblocks - 1
    '''
    H_b is the subblock of H that pertains only to basis (P,S) aka H_epsilon
    and is the first page in H_z.
    '''
    H_b = H_z[:, :, 0]

    H_b = H_b.astype('float64') 
    
    num_states = np.linalg.matrix_rank(H_b)

    # SVD decomposition of H_b subblock
    if num_states < Psize and num_states < Ssize:
        # H_b is NOT a full (row or column) rank matrix 
        # and we will use the sparse SVD decomp
        # sparse SVD decomp only works if NOT a full (row or column) rank matrix 
        U, SING, V = svds(H_b, k=num_states)
        P = U @ np.diag(SING)
        S = V
    else:
        # num_states = Psize or num_states = Ssize:
        # H_b is full row or column rank so we will use 
        # standard SVD decomp in numpy
        U, SING, V = svd(H_b)

        # we need to modify the original SING matrix
        # so the matrix multipliation works out
        
        # Create a matrix filled with zeros
        SING_FULL = np.zeros( (num_states, Ssize) )

        # Replace the diagonal elements in the result matrix with the values from the original matrix
        SING_FULL[:, :num_states] = np.diag(SING)

        P = U 
        S = SING_FULL @ V
    # U, SING, V = svd(H_b)
    # P = U @ np.diag(SING)
    # S = V
    B = P2Basis(P)

    Pnew = P @ np.linalg.pinv(B)
    Snew = B @ S
    
    # round to nearest integer. Each number in P, S should be close to 1 or zeero. 
    # This should aid computation
    Pnew = np.round( Pnew )
    Snew = np.round( Snew )
    T_array = np.empty( ( num_states, num_states, num_letters ) )


    #Take pseudo inverse of P,S Assuming Pnew and Snew are NumPy arrays

    P_plus = np.linalg.pinv(Pnew)
    S_plus = np.linalg.pinv(Snew)
    
    #round to nearest integer. Each number in Pplus, Splus should be close to 1 or zeero. 
    # This should aid computation
    # P_plus = np.round( P_plus )
    # S_plus = np.round( S_plus )

    for index in range( num_letters ):
        # indices used for filling T_array
        #index_array = index * n
        # indices used for selecting the right subblock in H_z
        #index_subblock = (index + 1) * Psize
        # selects subblock
        #H_subblock = H_z[index_subblock:index_subblock + Psize, :]
        H_subblock = H_z[:, :, index + 1] 
        #starts at the first subblock past the basis subblock
        # fills T_array
        T_array[:, :, index] = P_plus @ H_subblock @ S_plus

    t_0   = Pnew[0, :].reshape(1, -1) #maps a 1D aray to a row vector
    t_inf = Snew[:, 0].reshape(-1, 1) #maps a 1D aray to a column vector

    return t_0, T_array, t_inf, Pnew

def word2trans_matrix(word, T_array):
    """Function that outputs a transition matrix associated with a word for a given WFA
    
    INPUTS
    ------
    word: natural number numpy array or string of natural numbers 
        Word whose transition matrix we wish to evaluate. Must be ordered in the same
        way as T_array.
    T_array: (nxnxm) numpy array 
        contains transition matrices for each letter in the alphabet SIGMA. T_array = [A_w1| A_w2| ...| A_wm] where | denotes a new page.  Must be properly ordered.

    OUTPUTS
    -------
    T_word: (nxn) numpy array 
    ransition matrix defined by WFA for input word.
    n - # of states 
    m - # of letters
    """
    
    n, _, m   = T_array.shape
    len_word  = len(word)
    indicator = isinstance(word[0], str) 
    #tests for word as a string or numpy array
    
    T_word = np.eye(n)
    for ii in range(len_word):
        if indicator:
            letter_index = int(word[ii])
        else:
            letter_index = word[ii]

        if letter_index == 0:  # Case where the empty symbol is used.
            T_next = np.eye(n)
        else:
            T_next = T_array[:, :, letter_index-1]

        T_word = T_word @ T_next

    return T_word

def WFA_func(word, t_0, T_array, t_inf):
    """Function that outputs a weight associated with a word for a given WFA.
    
    DEPENDENCIES
    ------------
    word2trans_matrix
    
    INPUTS
    ------
    word: natural number numpy array or string of natural numbers Word whose transition matrix we wish to evaluate. Must be ordered in the same way as T_array.
        
    t_0 : 1xn row vector 
        initial weights of learned WFA
        
    T_array :  nxnxm numpy array 
        concatenation of transition matrices with size form [ A_x1| A_x2| ...| A_xm ]
        
    t_inf : nx1 column vector 
        Final weights of learned WFA      
        
    OUTPUTS
    -------
    T_word: (nxn) numpy array 
    ransition matrix defined by WFA for input word.
    
    n - # of states 
    m - # of letters
    """
    n, _, m = T_array.shape
    len_word = len(word)

    T_word = word2trans_matrix(word, T_array)
    f_word = t_0 @ T_word @ t_inf

    return f_word

def DFA2FST(t_0, T_array, t_inf, input_table, output_table, auto_key_list):
    """Function that maps a DFA to an FST
    
    DEPENDENCIES
    ------------
    pywrapfst as fst
    inverse_cantor_pairing
    cantor_pairing
    
    INPUTS
    ------
        
    t_0 : 1xn row vector numpy array
        initial weights of learned WFA
        
    T_array :  nxnxm numpy array 
        concatenation of transition matrices with size form [ A_x1| A_x2| ...| A_xm ]
        
    t_inf : nx1 column vector 
        Final weights of learned WFA      
        
    OUTPUTS
    -------
    f: fst object
        FST defined by the input transition matrices, initial and finals weights. 
        Each DFA letter is mapped to FST input and output symbol indices using the cantor cand inverse cantor maps. 
        
    FST_array: nxnxm numpy array
        Transition matrices ordered in the same way as T_array except rounded to the nearest integer

    n - # of states 
    m - # of letters
    """
    num_states, _,  num_letter_pairs = T_array.shape
    num_letter_pairs = int( num_letter_pairs )
    num_states       = int( num_states )
    FST_array  = np.rint( T_array )
    
    
    #extract initial and final states indices
    
    #only care about column indices of t_0 since t_0 is a 2D row vector
    _, start_state_index   = np.array( np.nonzero( t_0 > 0.6)   )
    #only care about row indices of t_inf since t_inf is a 2D column vector
    final_state_indices, _ = np.array( np.nonzero( t_inf > 0.6) )
    
    #THERE CANNOT BE MORE THAN 1 INITIAL STATE
    if start_state_index.size > 1:
        print("ERROR: More than 1 initial state in input Automaton")
    
    #now to create FST structure from FST_array
    
    # Create an empty FST with proper final states
    f = fst.Fst()
    for i in range(0, num_states): #add states to fst
        f.add_state()  
        if np.count_nonzero(final_state_indices == i) == 1: #checks to see if ith state is a final state.
            f.set_final( i )
    
    #add input and output symbol tables from given tables
    f.set_input_symbols( input_table )
    f.set_output_symbols( output_table )
    
    #all transitions share the identity weight
    identity_weight = fst.Weight.One(f.weight_type())
    # for every state in the FST we must iterate through each state and input/output message pair 
    # to determine if a transition is added
    # for start_index in range(0, num_states): #iterates through states 
    #     for end_index in range(0, num_states):
    #         for pair_index in range(0, num_letter_pairs):
    #             if FST_array[start_index, end_index, pair_index] == 1:
    #                 """now to determine proper keys and symbols 
    #                 to add to FST transition given from input and output tables""" 
    #                 auto_key = auto_key_list[pair_index + 1]
    #                 input_key, output_key = auto_key2fst_key( auto_key )
    #                 new_arc = fst.Arc(input_key, output_key, identity_weight, end_index)
    #                 f.add_arc(start_index, new_arc ) 
    possible_start_indices, possible_end_indices, possible_pair_index \
        = np.where( FST_array == 1 )
        
    # for every state in the FST we must iterate through each state and input/output message pair 
    #to determine if a transition is added
    num_possible_trans = possible_start_indices.size 
    
    for trans_index in np.arange(0, num_possible_trans):
        start_index = possible_start_indices[trans_index]
        pair_index  = possible_pair_index[trans_index]
        end_index   = possible_end_indices[trans_index]
        
        auto_key = auto_key_list[pair_index + 1]
        input_key, output_key = auto_key2fst_key( auto_key )
        new_arc = fst.Arc(input_key, output_key, identity_weight, end_index)
        f.add_arc(start_index, new_arc )  
        
     
    # Set the start state
    f.set_start(start_state_index)           
    return f, FST_array 

def mask2Hz_word(P, S, Alphabet):
    """Creates 3D array of words rperesenting the Hz matrix. FIRST ELEMENT IN EACH INPUT MUST BE 'EPSILON' OR''

    Args:
        P (list): First set of words in mask
        S (list): Ssecond set of words in mask
        Alphabet (list): Alphabet

    Returns:
        Hz (|P|x|S|x|Alphabet| size numpy array): 3D Array with a word at each Hz entry instead of the automaton binary vallue. 
    """
    #Replace first entry in P,S,Alphabet with ''
    P[0]        = ''
    S[0]        = ''
    Alphabet[0] = ''
    num_P    = len( P )
    num_S    = len( S )
    num_Alph = len( Alphabet )
    
    # initialize Hz_word 
    Hb_word =  [[p + s for s in S] for p in P]
    Hb_word_array = np.array(Hb_word, dtype = str) 
    Hz_word_array = Hb_word_array.copy()
    for letter_index in np.arange(1,num_Alph):
        letter = Alphabet[letter_index]
        Pnew   = [word + letter for word in P] 
        Hz_subblock = [[p + s for s in S] for p in Pnew]
        Hz_subblock_array = np.array( Hz_subblock, dtype = str )
        #Hz_word_array = np.concatenate(( Hz_word_array[:, :, np.newaxis] , Hz_subblock_array[:, :, np.newaxis] ), axis = 2 )
        if letter_index == 1:
            Hz_word_array = np.concatenate((Hz_word_array[:, :, np.newaxis], Hz_subblock_array[:, :, np.newaxis]), axis=2)
        else: 
            Hz_word_array = np.concatenate((Hz_word_array, Hz_subblock_array[:, :, np.newaxis]), axis=2)
    return Hz_word_array, Hb_word_array 

def alphabet2mask(Alphabet, max_word_length):
    """Creates a relevant mask of words (2 lists) based on alphabet given and max_word_length

    Args:
        Alphabet (list): alphabet of language, assumed to include epsilon in list but does not count
        max_word_length (Natural Number): _description_

    Returns:
        P (list): 
        S (list): 
    """
    if max_word_length < 1:
        return 'Error: max_word_lenght must be >= 1'
    else:
        Alphabet_meps = Alphabet[1:] #alphabbet minus the empty letter
        
        num_letters = len( Alphabet_meps ) 
        P = []
        for length_index in np.arange(1, max_word_length + 1):
            repeat = length_index  # Set the number of repetitions

            all_permutations = list( product(Alphabet_meps, repeat=repeat))

            #convert list to array
            perm_array  = np.array(all_permutations) 
            
            Padd = ''
            for comb_index in np.arange(0,length_index):
                Padd = np.core.defchararray.add( Padd, perm_array[:,comb_index] )
            P = np.hstack( (P, Padd) )
        P = np.insert(P, 0, 'epsilon')
        S = P.copy()
        return P, S 
    
def word_list2mask( word_list, Alphabet,  max_P_length, max_S_length):
    """Maps a list of words from a sampled automaton into a suitable mask

    Args:
        word_list (list of lists): Each list within word_list refers to the word generated by an individual automaton walk.
                                   word = word_list[ index ] = ['letter_1', letter_2', ...'letter_final] 
                                   each letter is a string
        Alphabet (list): contains each letter in the FST alphabet including "epsilon"
                         Note that Alphabet[0] will be reduced to '' (empty string) if it is not already
        max_P_length (integer): max word length in P
        max_S_length (integer): max word length in S
    Returns:
        P (list): list of squished words found to be accepted by automaton from word_list < max_P_length.   
        S (list): list of squished words found to be accepted by automaton from word_list < max_S_length.              
    """
    #remove word lists of length > max_P_length
    word_list_reduced = [sublist for sublist in word_list if len(sublist) <= max_P_length ]

    num_words = len( word_list_reduced )
    
    #Ensure alphabet has first element being empty string
    if not Alphabet[0] == '':
        Alphabet[0] == ''
    S = Alphabet.copy() 
    P = []
    for sample_index in np.arange(0, num_words):
        current_word_list = word_list_reduced[ sample_index ]
        #number of transitions taken by automaton during sample run
        num_transitions  = len( current_word_list )
        
        #all prefixes in word_list
        prefix_list = [ ''.join( current_word_list[:i+1] ) for i in np.arange(0, num_transitions ) ]
            
        for prefix_index in np.arange(0, num_transitions): 
            # prefix_index describes how many transitions the auto has made along current path 
            # that gave current_word_list as a word
            current_prefix = prefix_list[ prefix_index ]
            #check to see if current_prefix is in words_accept list already 
            if not current_prefix in P:
                P.append( current_prefix )
                # see if prefix belongs in S, it must have length <= max_S_length 
                # and not already be contained in S
                if num_transitions <= max_S_length and not current_prefix in S:
                    S.append( current_prefix )

    
    return P, S
    
def auto2Hz(auto, num_samples, max_length_sample, max_P_length, max_S_length ):
    """Samples automaton and generates an Hz matrix. 
    
    AUTO MUST BE AN AUTOMATON WITH MATCHING INPUT/OUTPUT SYMBOLS. 
    
    FUNCTION ASSUMES INPUT FST IS AN AUTOMATON AND ONLY INPUT SYMBOLS AND ALPHABET ARE USED

    Args:
        auto (mutable FST object): Automaton we wish to learn
        num_samples (natural number): how many times the automaton will be sampled (random walk #)
        max_length_sample (natural number): maximum # of transitions made by walk/letters in word (1 letter per transition)
        max_P_length (nautral number): maximum word length for possible words in mask sets
    Returns:
        Hz_array (3D numpy array): Array used to learn Automaton in spectral learning algorithm. Page size determined by sample length
        words_accept (list): list containing all accepted words by automaton (from word list) with no repeats
    """

    _, _, input_alphabet, output_alphabet, input_key_list, output_key_list\
    = fst2tables( auto )
    
    if input_alphabet == output_alphabet and input_key_list == output_key_list : 
        #case where input fst is an automaton as expected
        #Since the input symbols and output symbols are the same
        #the automaton symbols/letters are defined as the input symbols
        
        # otain random sample of words
        input_word_list, _, _,  \
        input_word_key_list, _, _ ,\
            _, _, _ \
                = fst2word_list(auto, max_length_sample, num_samples)
        num_unique_samples = len( input_word_list )        
        auto_alphabet = input_alphabet.copy()
        
        P, S = word_list2mask(input_word_list, Alphabet=auto_alphabet, max_P_length=max_P_length, max_S_length= max_S_length  ) 
        #obtain Hz_word_array   
        Hz_word_array, _ = mask2Hz_word(P, S, auto_alphabet)
        
        
        #initialize Hz_array with zeros to be equal in shape to Hz_word_array
        Hz_array = np.zeros_like( Hz_word_array, dtype=int )
        #Empty transition is allowed in initial state
        Hz_array[0,0,0] = 1
        #initialize list of accepted words by automaton as empty list
        words_accept = []
        
        
        for sample_index in np.arange(0, num_unique_samples):
            
            """Need to add code that 
            1. adds every prefix to Hz
            2. does not add words that have already appeared"""
            word_list        = input_word_list[ sample_index ]
            #number of transitions taken by automaton
            num_transitions  = len( word_list )
            #all prefixes in word_list
            word_prefix_list = [ ''.join( word_list[:i+1] ) for i in np.arange(0, num_transitions ) ]
            
            for prefix_index in np.arange(0, num_transitions): 
                current_prefix = word_prefix_list[ prefix_index ]
                #check to see if current_prefix is in words_accept list already 
                if not current_prefix in words_accept:
                    """current_prefix is not in the words_accept list and has not been recorded. 
                    We will add it to the list and change the relevant entries in Hz_word_array to 1 from 0"""
                    words_accept.append( current_prefix )
                    
                    #find where current prefix is in Hz_word_array with relevant indices
                    word_row_indices, word_column_indices, word_page_indices \
                        = np.where( Hz_word_array == current_prefix )
                    #next fill Hz_array with 1's along indices given by where function.
                    #change indices to 1 in Hz_array 
                    Hz_array[word_row_indices, word_column_indices, word_page_indices] = 1    
            
        return Hz_array, Hz_word_array, words_accept 
    else:
        #INPUT FST WAS NOT AN AUTOMATON
        print('Error: Input FST must be an automaton/acceptor')

def dfs(f, state, current_path, current_word, current_length, max_length, \
        paths_key, paths_letter, input_table):
    '''Depth First Search of f to provide all accepted words of f with length <= max_length
    
    Args:
    f - mutable fst object MUST BE AUTOMATON 
    (each transition has same input and output symbol)
    state - current state of path in search
    current_path - keys of words corresponding to current path taken (list)
    current_word - word corresponding to current path taken (list)
    current_length - length of word corresponding to current path taken
    max_length - describes maximum "depth" of dfs algorithm 
    paths_key - keys of accepted paths thus far (list of lists)
    paths_letter - words of accepted paths thus far (list of lists)
    input_table - input table of f(MUST EQUATE OUTPUT TABLE)

    Returns:
    paths_key - keys of accepted paths (list of lists)
    paths_letter - words of accepted paths (list of lists)

    '''
    if current_length > max_length:
        return
    #add current prefix to list
    if current_length > 0:  # Append the path if it's not empty
        paths_key.append(current_path[:])
        paths_letter.append(current_word[:])

    if current_length == max_length:  # Reached the maximum length, stop
        return
    for arc in f.arcs(state):
        if arc.ilabel == arc.olabel:  # Check if the input and output labels are the same (identity weight)
            next_state = arc.nextstate
            #add new letters/keys to current path
            next_path_key    = current_path + [arc.ilabel] #add to list of keys for current path
            input_letter  = input_table.find(arc.ilabel)
            next_path_letter = current_word + [input_letter] #add to list of paths for current path
            #recursive step
            dfs(f, next_state, next_path_key, next_path_letter, current_length + 1, max_length, \
                paths_key, paths_letter, input_table)
        else:
            return 'error: input must be an automaton'

def auto_paths(f, n):
    '''
    provide all accepted words of f with length <= n
    Args:
    f - mutable fst object MUST BE AUTOMATON 
    n - describes maximum "depth" of dfs algorithm 

    Returns: 
    paths_key - keys of accepted paths (list of lists)
    paths_letter - words of accepted paths (list of lists)
    '''

    #initialize inputs for depth first search algorithm
    start_state = f.start()
    paths_key = []
    paths_letter = []
    input_table  = f.input_symbols()
    dfs(f, start_state, [], [], 0, n, paths_key, paths_letter, input_table)
    return paths_key, paths_letter

def auto2basis(auto):
    """Maps automaton to a sufficient basis that can generate auto using SPEC_DECOMP

    Args:
    auto - fst object where each transition has the same input and output symbol
    Returns:
    P, S - basis
    alphabet - alphabet of auto
    paths_key - keys of accepted paths (list of lists)
    paths_word - words of accepted paths (list of lists)
    """
    #how much do we increase our search space
    num_states = auto.num_states()

    #all we need is S to be of length >= num_states so we can ensure Hb is of proper rank
    _, _, alphabet, _, _, _ = fst2tables( auto )
    # S_current = alphabet
    count = num_states
    # while len(S_current) <= num_states:
    #     _, S_current = alphabet2mask(alphabet, count)
    #     count += 1

    # count = 4
    # _, S_current = alphabet2mask(alphabet, 4)

    paths_key, paths_word = auto_paths(auto, num_states + count + 1)
    P_key  = [path for path in paths_key if len(path) <= num_states]
    P_word = [path for path in paths_word if len(path) <= num_states]
    S_word = [path for path in paths_word if len(path) <= count]
    P = [['epsilon']] + P_word
    S_current = [['epsilon']] + S_word
    return P, S_current, alphabet, paths_key, paths_word  

def auto2small_basis(auto):

    """
    Maps automaton to a reduced basis that can generate auto using SPEC_DECOMP
    This function takes output from auto2basis and reduces the basis
    by removing repeated rows in Hb and removes corresponding rows from P

    Args:
        auto - fst object where each transition has the same input and output symbol

    Returns:
        P, S - basis SQUISHED [emptystring, word1, word2 ...]
        paths_word_squish - list containing all words in squished form, 1D list
        Hb_array_reduced - reduced Hb matrix containging ones and zeros of size |P|x|S| 
        Hb_word_array_reduced - reduced Hb matrix containging words of size |P|x|S| 
        Hb_list - original array
        Hb_word_array - original array of words
    """

    P_big, S, alphabet, paths_key, paths_word = auto2basis( auto )

    P_squish = [''.join(inner_list_P) for inner_list_P in P_big]
    S_squish = [''.join(inner_list_S) for inner_list_S in S]
    paths_word_squish = [''.join(inner_list_path) for inner_list_path in paths_word]

    Hz_word_array, Hb_word_array = mask2Hz_word(P_squish, S_squish, Alphabet=alphabet)
    #compute Hb_array based on paths_word and P_big and S
    #initialize Hb_array with zeros to be equal in shape to Hb_word_array

    Hb_array = np.zeros_like( Hb_word_array, dtype=int )
    #Empty transition is allowed in initial state
    Hb_array[0,0] = 1

    # add ones in proper spot with paths_word
    for word in paths_word_squish:
        #find where current prefix is in Hz_word_array with relevant indices
        word_row_index, word_column_index \
            = np.where( Hb_word_array == word )
        # word_index = np.where( Hb_word_array == word )
        # word_row_index    = word_index[0] 
        # word_column_index = word_index[1]
        Hb_array[word_row_index, word_column_index] = 1

    # Initialize an empty list to store indices of removed rows
    removed_indices = []

    # Convert list of lists to set of tuples to remove duplicates, then back to list of lists
    unique_rows = []
    Hb_word_reduced = []
    #temporarily convert Hb_array to a list
    Hb_list = Hb_array.copy()
    Hb_list.tolist()

    seen = set()
    for i, row in enumerate(Hb_list):
        row_tuple = tuple(row)
        if row_tuple not in seen:
            row_words = Hb_word_array[i, :]
            unique_rows.append(row)
            Hb_word_reduced.append( row_words )
            seen.add(row_tuple)
        else:
            removed_indices.append(i)
            #P_squish.pop(i)
    # Remove items corresponding to the indices
    P_small = [item for index, item in enumerate(P_squish) if index not in removed_indices]
    Hb_array_reduced = np.array( unique_rows )
    Hb_word_array_reduced = np.array( Hb_word_reduced )

    return P_small, S_squish, paths_word_squish, Hb_array_reduced, Hb_word_array_reduced, Hb_list, Hb_word_array  

def auto2Hz2(auto, method=None):
    """Generates an Hz matrix using auto2smallbasis. 
    
    AUTO MUST BE AN AUTOMATON WITH MATCHING INPUT/OUTPUT SYMBOLS. 
    
    FUNCTION ASSUMES INPUT FST IS AN AUTOMATON AND ONLY INPUT SYMBOLS AND ALPHABET ARE USED

    Args:
        auto (mutable FST object): Automaton we wish to learn
        method - indicator for whether auto2small_basis function is used or auto2basis is used
            method == 'none' then auto2small_basis is used
    Returns:
        Hz_array (3D numpy array): Array used to learn Automaton in spectral learning algorithm. Page size determined by sample length
        words_accept (list): list containing all accepted words by automaton (from word list) with no repeats
    """

    _, _, input_alphabet, output_alphabet, input_key_list, output_key_list\
    = fst2tables( auto )
    
    if input_alphabet == output_alphabet and input_key_list == output_key_list : 
        #case where input fst is an automaton as expected
        #Since the input symbols and output symbols are the same
        #the automaton symbols/letters are defined as the input symbols
        
        # otain random sample of words
        # input_word_list, _, _,  \
        # input_word_key_list, _, _ ,\
        #     _, _, _ \
        #         = fst2word_list(auto, max_length_sample, num_samples)
        # num_unique_samples = len( input_word_list )        
        auto_alphabet = input_alphabet.copy()
        if method: 
        # OPTION 2
            P_big, S, alphabet, paths_key, paths_word = auto2basis( auto )
            P_squish  = [''.join(inner_list_P) for inner_list_P in P_big]
            S_squish  = [''.join(inner_list_S) for inner_list_S in S]
            word_list = [''.join(inner_list_path) for inner_list_path in paths_word]
            # num_unique_samples = len( word_list )   
        else:
            # OPTION 1
            P_squish, S_squish, word_list, Hb_word, Hb_array, Hb_list, Hb_word_big \
                = auto2small_basis(auto)            
        
        # obtain Hz_word_array
        Hz_word_array, _ = mask2Hz_word(P_squish, S_squish, auto_alphabet)
        
        
        #initialize Hz_array with zeros to be equal in shape to Hz_word_array
        Hz_array = np.zeros_like( Hz_word_array, dtype=int )
        #Empty transition is allowed in initial state
        Hz_array[0,0,0] = 1
        #initialize list of accepted words by automaton as empty list
        
        
        for word in word_list:
            word_row_indices, word_column_indices, word_page_indices \
                = np.where( Hz_word_array == word )
            #next fill Hz_array with 1's along indices given by where function.
            #change indices to 1 in Hz_array 
            Hz_array[word_row_indices, word_column_indices, word_page_indices] = 1    
            
        return Hz_array, Hz_word_array, word_list
    else:
        #INPUT FST WAS NOT AN AUTOMATON
        print('Error: Input FST must be an automaton/acceptor')