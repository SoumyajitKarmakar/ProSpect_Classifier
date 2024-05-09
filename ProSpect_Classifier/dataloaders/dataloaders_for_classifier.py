from dataloaders.dataloaders import SetDataManager


class Dataloader_classifier:
    def __init__(self,image_array,pil_images,labels, data_len= None) -> None:
        pil_images = [x for sub_list in pil_images for x in sub_list]
        image_array = image_array.view(-1,*image_array.size()[2:])
        labels=labels.view(-1)
        
        self.pil_images = pil_images        ## list of images 
        self.image_array = image_array      ### tensor of images_arrays 
        self.labels = labels            ### tensor of the labels 
        
        
    def __getitem__(self,index):
        return self.pil_images[index], self.image_array[index], self.labels[index]
        
        
    def __len__(self):
        return len(self.labels)
        