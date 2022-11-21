import os 

dataset = 'office-caltech'

if dataset == 'office':
	domains = ['amazon', 'dslr', 'webcam']
elif dataset == 'office-caltech':
	domains = ['amazon', 'dslr', 'webcam', 'caltech']
elif dataset == 'office-home':
	domains = ['Art', 'Clipart', 'Product', 'Real_World']
else:
	print('No such dataset exists!')

for domain in domains:
	log = open('/home/zhongyi/data/domain_adaptation/classification/office-caltech'+'/'+domain+'_list.txt','w')
	directory = os.path.join('/home/zhongyi/data/domain_adaptation/classification/office-caltech', domain)
	classes = [x[0] for x in os.walk(directory)]
	classes = classes[1:]
	classes.sort()
	for idx,f in enumerate(classes):
		files = os.listdir(f)
		for file in files:
			s = os.path.abspath(os.path.join(f,file)) + ' ' + str(idx) + '\n'
			log.write(s)
	log.close()