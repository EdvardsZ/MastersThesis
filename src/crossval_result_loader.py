import torch
import os

class Loss:
    def __init__(self, loss_name: str, average: float, std: float):
        self.loss_name = loss_name
        self.average = average
        self.std = std    

class Result:
    def __init__(self, filename: str, data: dict[str, Loss]):
        self.filename = filename
        self.data = data
        
    def get_config_number(self) -> int:
        bracket_text = self.filename.split("(")[1].split(")")[0]
        num_embeddings = bracket_text.split("_")[0]
        embedding_dim = bracket_text.split("_")[1]
        hidden_dim = bracket_text.split("_")[2]
        num_blocks = bracket_text.split("_")[3]
        
        if embedding_dim == "16":
            return 1
        if embedding_dim == "32":
            return 2
        if embedding_dim == "64":
            return 3
        else :
            raise Exception("Unknown embedding_dim")
        
    
        
def get_crossval_results(data: list)-> dict[str, Loss]:
    losses: dict[str, Loss]= {}
    
    first : dict = data[0][0]
    
    keys = first.keys()
    
    for key in keys:
        values = []
        for item in data:
            values.append(item[0][key])
            
        average = sum(values) / len(values)
        std = sum([(value - average)**2 for value in values]) / len(values)
        
        losses[key] = Loss(key, average, std)
        
    return losses

def get_results(directory: str) -> list[Result]:
    results = []
    print("Loading results...")
    for filename in os.listdir(directory):
        if filename.endswith(".pt"):
            # load the file
            data = torch.load(directory + filename)
            data = get_crossval_results(data)
            results.append(Result(filename, data))
            #print(f"Loaded {filename}")
            
    print("Loaded: ", len(results))
    print("Done.")
    return results
    