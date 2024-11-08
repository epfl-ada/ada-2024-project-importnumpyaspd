import pandas as pd

def csv_loader(filename, repertory = 'IMDb', sep='\t', column_names=None, header=None):
    
    #plus tard changer le path car les loader seront plus dans ce répertoire
    if repertory == 'IMDb':
        path = './IMDb/'+filename+'.tsv'
    elif repertory == 'CMU':
        path = './CMU/'+filename+'.tsv'
    else : 
        raise ValueError("Unknown repertory, please enter a valid repertory name")
        
    if column_names == None:
        df = pd.read_csv(path, sep=sep, header=header,low_memory=False)
    else :
        df = pd.read_csv(path, sep=sep, header=header, names = column_names,low_memory=False)
        
    return df

#def pickle_loader():