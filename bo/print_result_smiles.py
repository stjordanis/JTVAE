def print_result_smiles(save_dir='results/'):
    import gzip
    import pickle
    import rdkit.Chem as Chem
    from rdkit.Chem import Draw
    from rdkit.Chem import Descriptors
    import sascorer
    def save_object(obj, filename):
        result = pickle.dumps(obj)
        with gzip.GzipFile(filename, 'wb') as dest: dest.write(result)
        dest.close()

    def load_object(filename):
        with gzip.GzipFile(filename, 'rb') as source: result = source.read()
        ret = pickle.loads(result)
        source.close()
        return ret
    j=0
    all_smiles = []
    fn = save_dir+'scores%d.dat' % (j)
    scores = load_object(fn)
    fn = save_dir+'valid_smiles%d.dat' % (j)
    smiles = load_object(fn)
    fn=save_dir+'valid_mols%d.png' % (j)
    valid_mols=load_object(fn)
    all_smiles.extend(zip(smiles, scores,valid_mols))
    all_smiles = [(x,-y,z) for x,y,z in all_smiles]
    all_smiles = sorted(all_smiles, key=lambda x:x[1], reverse=True)

    for s,v,m in all_smiles[0:5]:
        print s,v,m

#print_result_smiles()
