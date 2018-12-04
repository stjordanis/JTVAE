def print_result_demo(save_dir):
    import sys
    sys.path.append('/home/icml18-jtnn')
    import base64
    import gzip
    import pickle
    import rdkit.Chem as Chem
    from rdkit.Chem import Draw
    from rdkit.Chem import Descriptors
    import sascorer
    from PIL import Image
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
   # import cv2
    def save_object(obj, filename):
        result = pickle.dumps(obj)
        with gzip.GzipFile(filename, 'wb') as dest: dest.write(result)
        dest.close()

    def load_object(filename):
        with gzip.GzipFile(filename, 'rb') as source: result = source.read()
        ret = pickle.loads(result)
        source.close()
        return ret
    
    all_smiles = []
    for i in xrange(1,2):
        for j in xrange(5):
            fn = save_dir+'scores%d.dat' % (j)
            scores = load_object(fn)
            fn = save_dir+'valid_smiles%d.dat' % (j)
            smiles = load_object(fn)
            all_smiles.extend(zip(smiles, scores))
    size=(50,50)
    all_smiles = [(x,-y) for x,y in all_smiles]
    all_smiles = sorted(all_smiles, key=lambda x:x[1], reverse=True)
  
    for s,v in all_smiles:
	print s,v
        mol=Chem.MolFromSmiles(s)
        fig=Draw.MolToFile(mol,'mol.png')
	with open('mol.png','rb') as image_file:
		encoded_string=base64.b64encode(image_file.read())
	encoded_images.append(encoded_string)

        #cv2.waitKey(0)
        #	mol=Chem.MolFromSmiles(s)
       # fig=Draw.MolToMPL(mol,size=size)
#	fig=Draw.MolToFile(imol,'mol1.png')
  #      img=mpimg.imread('mol1.png')
 #       imgplot = plt.imshow(img)
#        plt.show()


       # image=Image.open('mol1.png')
       # image.show()
        
#print_result_demo('results/')
 #   mols = [Chem.MolFromSmiles(s) for s,_ in all_smiles[:3]]
#    vals = ["%.2f" % y for _,y in all_smiles[:3]]
  #  img = Draw.MolsToGridImage(mols, molsPerRow=1, subImgSize=(200,135), legends=vals, useSVG=True)
#print_result_demo('results/')
