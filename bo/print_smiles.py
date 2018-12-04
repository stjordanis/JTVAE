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
    j=4
    all_smiles = []
    fn = save_dir+'scores%d.dat' % (j)
    scores = load_object(fn)
    fn = save_dir+'valid_smiles%d.dat' % (j)
    smiles = load_object(fn)
    all_smiles.extend(zip(smiles, scores))
    all_smiles = [(x,-y) for x,y in all_smiles]
    all_smiles = sorted(all_smiles, key=lambda x:x[1], reverse=True)

    for s,v in all_smiles[0:5]:
        print s,v

print_result_smiles()
