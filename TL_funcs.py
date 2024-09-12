
import pywrapfst as fst
import numpy as np
import random 

def fst_key2auto_key(input_key, output_key):
    """_summary_

    Args:
        input_key (_type_): _description_
        output_key (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    key_array   = np.array([input_key, output_key])
    #convert key temporarily to string to combine indices
    key_str = ''.join(map(str, key_array))

    # Convert the string to an integer
    auto_key = int(key_str)
    
    return auto_key 

def cs_key2cs_label(cs_key):
    """Maps current state key (integer) to current state label (string)
        1 - H, 2 - B, 3 - F

    Args:
        cs_key (integer): current state key

    Returns:
        cs_label (string): current state label
    """
    
    if cs_key == 1:
        cs_label = 'B'
        
    if cs_key == 2:
        cs_label = 'H'
        
    if cs_key == 3:
        cs_label = 'F'
    return cs_label

def key2label(key):
    """ 
    output symbol labels
    (highway light coloe-farm road light color)
    rr - 1
    gr - 2
    rg - 3

    input symbol labels (4 binary marks
    (emergency indicator-highway car indicator, farm road car indicator, current state)
    emergency indicator     -- 0,1   == 1,2
    highway car indicator   -- 0,1   == 1,2
    farm road car indicator -- 0,1   == 1,2
    current state           -- B,H,F == 1,2,3
    """
    emergency_ind = np.array2string( key[0] - 1 )
    hcar_ind      = np.array2string( key[1] - 1 )
    fcar_ind      = np.array2string( key[2] - 1 )
    
    cs_key        = key[3]
    cs_label      = cs_key2cs_label( cs_key )
    
    label = emergency_ind + hcar_ind +fcar_ind + cs_label
    return label 

def output_key2output_label( output_key ):
    """Maps output key to output label

    Args:
        output_key (integer): 1 -rr, 2 - gr, 3 - rg
    """
    
    if output_key == 1:
        output_label = 'rr'
        
    if output_key == 2:
        output_label = 'gr'
        
    if output_key == 3:
        output_label = 'rg'
    return output_label

def fst_table2auto_table(input_symbols, output_symbols ):
    """concatenates input and output symbol tables of an FST into an automaton symbol table where the symbols are joined. e.g. (i,o) --> (io). The keys of each table are combined as well
    [1,i], [2,o] -->[12, io]
    Args:
        input_symbols  (pywrapfst object symbol table): input symbol table of fst we wish to concatenate with output_symbols into an automaton table
        output_symbols (pywrapfst object symbol table): output symbol table of fst we wish to concatenate with input_symbols into an automaton table

    Returns:
        autoo_symbol_table (pywrapfst object symbol table): Concatenated table
    """

    #Iterate through the symbols in each symbol table

    auto_symbol_table = fst.SymbolTable()
    for input_pair in input_symbols:
        for output_pair in output_symbols:
        
            input_key   = input_pair[0]
            input_label = input_pair[1]
        
            output_key   = output_pair[0]
            output_label = output_pair[1]

            current_key   = np.array([input_key, output_key])
            current_label = input_label + output_label
            #convert key temporarily to string to combine indices
            key_str = ''.join(map(str, current_key))

            # Convert the string to an integer
            current_key_int = int(key_str)
            # add concatenated keys and symbol to table
            auto_symbol_table.add_symbol( current_label , key = current_key_int)
        
        
    return auto_symbol_table 

def fst2auto(f, input_table = None, output_table = None):
    """_summary_

    Args:
        f (pywrapfst fst object): _description_
        input_table (pywrapfst table object): optional input symbols for automaton, fst input symbols are used if none are given
        output_table (pywrapfst table object): optional output symbols for automaton, fst output symbols are used if none are given

    Returns:
        auto (pywrapfst automaton): Automaton version of f. 
        if (s, i, o, s') is a transition in f then (s, io, io, s') is a transition 
        The keys in symbol table for auto is defined by concatenating the keys e.g.
        f_key(i) = 1, f_key(o) = 2 then auto_key(io) = 12
    """
    
    
    if input_table: 
        #case where automaton has different input symbols than original fst
        input_symbols = input_table 
    else: 
        #case where automaton has same in0.put symbols than original fst
        input_symbols  = f.input_symbols()
        
    if output_table: 
        #case where automaton has different output symbols than original fst
        output_symbols = output_table 
    else: 
        #case where automaton has same output symbols than original fst
        output_symbols  = f.output_symbols()
        
  
    
    auto_symbol_table = fst_table2auto_table(input_symbols, output_symbols )
    auto = fst.Fst() 
    
    auto.set_input_symbols(  auto_symbol_table )
    auto.set_output_symbols( auto_symbol_table )
    
    for state in f.states():
        auto.add_state()
        #all states are possible final states
        auto.set_final( state )
        
    for state in f.states():
        source_state = state
        for arc in f.arcs(state):
            input_key  = arc.ilabel
            output_key = arc.olabel
            auto_label_array   = np.array([input_key, output_key])
            #convert key temporarily to string to combine indices
            key_str = ''.join( map( str, auto_label_array) )

            # Convert the auto key string to an integer 
            # since keys for a mutable fst table must be integers
            current_key_int = int(key_str)
        
            weight = arc.weight
            destination_state = arc.nextstate
            
            #auto already has concatenated auto_table as a feature from fst_table2auto_table
            new_arc = fst.Arc(current_key_int, current_key_int, weight, destination_state)
            
            # Add the extracted arc information to the destination FST
            auto.add_arc(source_state, new_arc)
    # Set the start and final states of the destination FST if needed
    auto.set_start(f.start())
   
    
    return auto 

def TL_key2label_auto(key_auto):
    """Maps the concatenation of input and output keys into a single label

    Args:
        key_auto (5x1 numpy array): Concatenation of FST input and output keys 
    """
    input_key   = key_auto[0:4]
    output_key  = key_auto[4]
    
    input_label  = key2label( input_key )
    output_label = output_key2output_label(output_key)
    
    label_auto = input_label + output_label
    return label_auto

def TL_auto_syms_table():
    """Provides FST table object and numpy array containing a key for each symbol

    Returns:
        TL_syms_table (FST symbol table): object containing keys and labels for Traffic Light simulation
        TL_key_array (numpy array): array containing a list of all keys for TL symbols
    """
    # Create an automaton symbol table
    TL_syms_table = fst.SymbolTable()
    # create automata representing desired language 
    # Add symbols with integer IDs and corresponding strings
    # no cars on farm road, highway or emergency

    #creates automata symbol and key table where inputs and output for FSTs are concatenated
    TL_key_array = np.array([]) #initialize empty table
    for emer_key in range(1,3):
        for hcar_key in range(1,3):
            for fcar_key in range(1,3):
                for cs_key in range(1,4):
                    for output_key in range(1,4):
                        current_key   = np.array([emer_key, hcar_key, fcar_key, cs_key, output_key])
                        current_label = TL_key2label_auto(current_key)
                        #convert key temporarily to string to combine indices
                        key_str = ''.join(map(str, current_key))

                        # Convert the string to an integer
                        current_key_int = int(key_str)
                        TL_key_array    = np.hstack( (TL_key_array, current_key_int) )
                        TL_syms_table.add_symbol( current_label , key = current_key_int)
        
    return TL_syms_table, TL_key_array 

def TL_input_syms_table():
    """Input symbols for traffic light simulation

    Returns:
        TL_input_table (_type_): FST table object containing labels and keys for input symbols
        TL_input_key_array (numpy array): Array containing all keys in the input table
    """
    # Create an empty input symbol table and empty key array
    TL_input_table     = fst.SymbolTable()
    TL_input_key_array = np.array([])
    
    # add empty letter to table and key list
    #TL_input_table.add_symbol( 'epsilon' , key = 0)
    # Add symbols with integer IDs and corresponding strings

    for emer_key in range(1,3):
        for hcar_key in range(1,3):
            for fcar_key in range(1,3):
                for cs_key in range(1,4):
                    current_key   = np.array([emer_key, hcar_key, fcar_key, cs_key])
                    current_label = key2label(current_key)
                    #convert key temporarily to string to combine indices
                    key_str = ''.join(map(str, current_key))

                    # Convert the string to an integer
                    current_key_int = int(key_str)
                    # add key to array list
                    TL_input_key_array = np.hstack( (TL_input_key_array, current_key_int) )
                    TL_input_table.add_symbol( current_label , key = current_key_int)
    
    return TL_input_table, TL_input_key_array
   
def TL_output_syms_table():
    """output symbols for traffic light simulation

    Returns:
        TL_output_table (_type_): FST table object containing labels and keys for output symbols
        TL_output_key_array (numpy array): Array containing all keys in the output table
    """
    # Create an empty output symbol table and empty key array
    TL_output_table     = fst.SymbolTable()
    TL_output_key_array = np.array([1, 2, 3])
    TL_output_table.add_symbol('rr', key=1)
    TL_output_table.add_symbol('gr', key=2)
    TL_output_table.add_symbol('rg', key=3)
    
    return TL_output_table, TL_output_key_array   
   
def TL_desired_lang():  
    """Creates FST representation of desired language for TL simulation

    Returns:
        desired_lang (mutable fst): desired language of traffic light 
    """
    # Create an empty FST
    desired_lang = fst.Fst()
    
    input_syms, _  = TL_input_syms_table()
    output_syms, _ = TL_output_syms_table()
    
    #add input and output symbol tables
    desired_lang.set_input_symbols(input_syms)
    desired_lang.set_output_symbols(output_syms)

    #add highway green, both red and farm green states
    H = desired_lang.add_state()
    B = desired_lang.add_state()
    F = desired_lang.add_state()

    #starts with both lights red
    desired_lang.set_start( B )
    
    #all states are possible final states
    desired_lang.set_final( H )
    desired_lang.set_final( B )
    desired_lang.set_final( F )

    #identity weight 
    eye_weight = fst.Weight.One(desired_lang.weight_type())

    #add stationary transitions for highway green
    desired_lang.add_arc(H, fst.Arc(1112, 2, eye_weight, H) )
    desired_lang.add_arc(H, fst.Arc(1212, 2, eye_weight, H) )

    #add stationary transitions for both red 
    desired_lang.add_arc(B, fst.Arc(2111, 1, eye_weight, B) )
    desired_lang.add_arc(B, fst.Arc(2121, 2, eye_weight, B) )
    desired_lang.add_arc(B, fst.Arc(2211, 1, eye_weight, B) )
    desired_lang.add_arc(B, fst.Arc(2221, 2, eye_weight, B) )

    #add stationary transitions for farm road green
    desired_lang.add_arc(F, fst.Arc(1113, 3, eye_weight, F) )
    desired_lang.add_arc(F, fst.Arc(1123, 3, eye_weight, F) )

    #add transition both red to highway green
    desired_lang.add_arc(B, fst.Arc(1211, 2, eye_weight, H) )
    desired_lang.add_arc(B, fst.Arc(1221, 2, eye_weight, H) )

    #add transtition both red to farm green
    desired_lang.add_arc(B, fst.Arc(1121, 3, eye_weight, F) )
    desired_lang.add_arc(B, fst.Arc(1221, 3, eye_weight, F) )

    #add transition highway green to both red 
    #emergency vehicle present
    desired_lang.add_arc(H, fst.Arc(2112, 1, eye_weight, B) )
    desired_lang.add_arc(H, fst.Arc(2122, 1, eye_weight, B) )
    desired_lang.add_arc(H, fst.Arc(2212, 1, eye_weight, B) )
    desired_lang.add_arc(H, fst.Arc(2222, 1, eye_weight, B) )

    #no emergency vehicle present
    desired_lang.add_arc(H, fst.Arc(1122, 1, eye_weight, B) )
    desired_lang.add_arc(H, fst.Arc(1222, 1, eye_weight, B) )

    #add transition farm road green to both red 
    #emergency vehicle present
    desired_lang.add_arc(F, fst.Arc(2113, 1, eye_weight, B) )
    desired_lang.add_arc(F, fst.Arc(2123, 1, eye_weight, B) )
    desired_lang.add_arc(F, fst.Arc(2213, 1, eye_weight, B) )
    desired_lang.add_arc(F, fst.Arc(2223, 1, eye_weight, B) )

    #no emergency vehicle present
    desired_lang.add_arc(F, fst.Arc(1213, 1, eye_weight, B) )
    desired_lang.add_arc(F, fst.Arc(1223, 1, eye_weight, B) )

    return desired_lang 
   
def periodic_attacker(output_syms_table, output_syms_key, period):
    """Creates a periodic deterministic attacker FST

    Args:
        output_syms_table (fst table): output symbol table for plant
        period (integer): how often an attack occurs. More precisely how many timesteps between attacks
    returns:
        attack (mutable FST): deterministic attacker with given period  
    """
    num_states  = period + 1
    num_symbols = output_syms_key.size 
    
    # Create an empty FST with proper final states
    attack = fst.Fst()
    
    #add input and output symbols 
    attack.set_input_symbols( output_syms_table )
    attack.set_output_symbols( output_syms_table )
     
    for i in range(0, num_states): #add states to fst
        attack.add_state()  
        attack.set_final( i ) #all states are possible final states
        
    attack.set_start( 0 )    
    #all transitions share the identity weight
    eye_weight = fst.Weight.One(attack.weight_type())   
    
    #add inactive transitions 
    # the number of inactive transitions = period of attack
    for inactive_trans_index in range(0,period):
        for syms_index in output_syms_key:
            beg_state = inactive_trans_index
            end_state = beg_state + 1
            new_arc   = fst.Arc(syms_index, syms_index, eye_weight, end_state)
            attack.add_arc(beg_state, new_arc)    
    
    
    #now add manipulated transition 
    shuffle_key_array = np.random.permutation(output_syms_key)
    for index in range( 0, num_symbols):
        #shuffled transitions occur between final state and initial state 
        arc_shuffle = fst.Arc( output_syms_key[index], shuffle_key_array[index], eye_weight, 0)
        attack.add_arc(num_states-1, arc_shuffle)
    return attack 

def fst2word(f, max_word_length, seed = None ): 
    """Creates a random word accepted by fst. 

    Args:
        f (mutable fst): fst we wish to sample
        max_word_length (natural numebr): maximum length of word

    Returns:
        input_word (list): input word from the random walk
        input_word_key (list): keys of each input letter in input word in same order as word
        output_word (list): output word from the random walk
        output_word_key (list): keys of each input letter in output word in same order as word
        auto_word (list): automaton word from the random walk with both input and output
        auto_word_key (list): keys of each auto word same order as word
        input_alphabet (list): list of input symbols in the alphabet
        output_alphabet (list): list of output symbols in the alphabet
        auto_alphabet (list): list of automaton symbols in the alphabet
    """
    # Set the random seed if not provided
    if seed is None:
        rng = np.random.default_rng()
        seed = rng.integers(low=0, high=1e6)  
        

    #sort the fst for repeatability and uniformity
    f = f.arcsort()
    
    #pull input output symbol tables from fst create auto table from input/output tables
    input_table  = f.input_symbols()
    output_table = f.output_symbols()
    auto_table = fst_table2auto_table( input_table, output_table )
    
    #initialize alphabets as lists containing only epsilon denoting the empty symbol 
    input_alphabet  = ["epsilon"]
    output_alphabet = ["epsilon"]
    auto_alphabet   = ["epsilon"]
    
    #creates input, output alphabet lists
    #iterate through all input and output letters
    
    #if have time make this more efficient
    for input_pair in input_table:
         #input pair of letters and keys
        input_key    = input_pair[0]
        input_letter = input_pair[1]
        #add to each input alphabet 
        input_alphabet.append( input_letter )
        
    for output_pair in output_table:
        #output pair of letters and keys
        output_key    = output_pair[0]
        output_letter = output_pair[1]
            
        #add to each output alphabet 
        output_alphabet.append( output_letter )
            
    #create alphabet for automaton
    for auto_pair in auto_table:
        auto_letter   = auto_pair[1]
        auto_alphabet.append( auto_letter )
        
    
    # initialize empty lists
    input_word      = []
    input_word_key  = []
    output_word     = []
    output_word_key = []
    auto_word       = []
    auto_word_key   = []
    
    #create random walk through fst 
    #randgen only generates paths with num_transitions < max_word_length so we add 1 in order to get paths with
    # that have num_transitions = max_word_length
    random_fst = fst.randgen(f, max_length = max_word_length+1, seed = seed, select = 'uniform' )
    # Iterate over the arcs in the generated FST
    for state in random_fst.states():
        for arc in random_fst.arcs(state):
            
            input_key  = arc.ilabel
            output_key = arc.olabel
            auto_key   = fst_key2auto_key(input_key, output_key )
            
            #now to find letters associated with each key 
            input_letter  = input_table.find(input_key)
            output_letter = output_table.find(output_key)
            auto_letter   = auto_table.find(auto_key)
            
            input_word.append( input_letter )
            output_word.append( output_letter )
            auto_word.append( auto_letter )
            
            input_word_key.append( input_key )
            output_word_key.append( output_key )
            auto_word_key.append( auto_key )
            
    return input_word, output_word, auto_word, \
        input_word_key, output_word_key, auto_word_key, \
            input_alphabet, output_alphabet, auto_alphabet  

def fst2tables( f ):
    
    
    #pull input output symbol tables from fst create auto table from input/output tables
    input_table  = f.input_symbols()
    output_table = f.output_symbols()
    
    input_alphabet  = ["epsilon"]
    output_alphabet = ["epsilon"]
    auto_alphabet   = ["epsilon"]
    input_key_list  = [0]
    output_key_list = [0]
    #creates input, output alphabet lists
    #iterate through all input and output letters
    
    #if have time make this more efficient
    for input_pair in input_table:
         #input pair of letters and keys
        input_key    = input_pair[0]
        input_letter = input_pair[1]
        #add to each input alphabet and key list
        input_alphabet.append( input_letter )
        input_key_list.append( input_key)
        
    for output_pair in output_table:
        #output pair of letters and keys
        output_key    = output_pair[0]
        output_letter = output_pair[1]
            
        #add to each output alphabet and key list
        output_alphabet.append( output_letter )
        output_key_list.append( output_key )
    
    return input_table, output_table, \
        input_alphabet, output_alphabet, \
        input_key_list, output_key_list 

def auto_key2fst_key( auto_key ):
    """_summary_

    Args:
        auto_key (_type_): _description_

    Returns:
        _type_: _description_
    """
    auto_key_str = str( auto_key )
    if  len( auto_key_str )% 2 == 0:
        #len( auto_key ) is divisible by 2
        
        format_size = int( len( auto_key_str )/2 )
        input_key_str  = auto_key_str[0:format_size]
        output_key_str = auto_key_str[format_size:]
        input_key_int  = int( input_key_str )
        output_key_int = int( output_key_str )
        return input_key_int, output_key_int 
    else: 
        print('ERROR: Auto Key must have format divisble by 2', auto_key_str)
    
def print_fst(f):
    input_table  = f.input_symbols()
    output_table = f.output_symbols()
    for state in f.states():
        print(f"State {state}:")
        for arc in f.arcs(state):
            input_letter   = input_table.find(arc.ilabel)
            output_letter  = output_table.find(arc.olabel)
            print(f"  Transition: {input_letter} -> {output_letter}, Next State: {arc.nextstate}")
            
def fst2word_list(f, max_word_length, num_words, seed = None):
    """Generate random list of words accepted by f

    Args:
        f (_type_): _description_
        max_word_length (_type_): _description_
        num_words (_type_): _description_

    Returns:
        input_word_list (multidimensional list): _description_
        output_word_list (multidimensional list): _description_
        auto_word_list (multidimensional list): _description_
    """
    
    
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
        
    input_word_list      = []
    input_word_key_list  = []
    output_word_list     = []
    output_word_key_list = []
    auto_word_list       = []
    auto_word_key_list   = []
    
    #extract input, output, and auto alphabets
    _, _, _, \
        _, _, _, \
            input_alphabet, output_alphabet, auto_alphabet \
                = fst2word(f, max_word_length, seed)
    
    for index in np.arange(0,num_words):
        input_word, output_word, auto_word, \
            input_word_key, output_word_key, auto_word_key, \
                _, _, _= fst2word(f, max_word_length, seed)
        if not auto_word in auto_word_list:   
            input_word_list.append ( input_word )
            output_word_list.append( output_word )
            auto_word_list.append( auto_word )
            
            input_word_key_list.append ( input_word_key )
            output_word_key_list.append( output_word_key )
            auto_word_key_list.append( auto_word_key )
        # Increment the seed for the next iteration to get a different random sequence
        seed += 1
    
    return input_word_list, output_word_list, auto_word_list,  \
        input_word_key_list, output_word_key_list, auto_word_key_list, \
            input_alphabet, output_alphabet, auto_alphabet 
