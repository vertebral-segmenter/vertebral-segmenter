import synapseclient 
import synapseutils 

syn = synapseclient.Synapse() 
syn.login('vertebral.segmenter','csc413-research')

data_path = "./dataset"
files = synapseutils.syncFromSynapse(syn, entity='syn3193805', path=data_path)
