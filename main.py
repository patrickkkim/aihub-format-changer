from PIL import Image
from random import sample

import xml.etree.cElementTree as ET
import multiprocessing
import ray, psutil
import shutil
import glob
import json
import os


class ProgressLoader:
    def __init__(self):
        self.progress = 0
        self.length = 1
        self.subProgress = 0
        self.subLength = 1
        self.blockCount = 50

        self.printWhole("Waiting for process")

    def printIncrement(self, i, subtext="", speed=7):
        if i % int(self.subLength / speed) == 0:
            self.printWhole(subtext)

    def printWhole(self, subtext=""):
        self.clear()
        self.printProgress(self.progress, self.length, "Total: ")
        self.printProgress(self.subProgress, self.subLength, subtext)

    def printProgress(self, progress, length, text=""):
        progressPercentage = (progress / length * 100.0)
        count = int(progressPercentage / 100.0 * self.blockCount)
        
        progressBar = "█" * count + "░" * (self.blockCount - count)
        print(progressBar)
        text = str(text)
        print(text + " {:.1f}%...".format(progressPercentage))
        print("\n")

    def updateLength(self, length, increment=False):
        if increment: self.length += 1
        else: self.length = length

    def updateSubLength(self, length, increment=False):
        if increment: self.subLength += 1
        else: self.subLength = length

    def updateProgress(self, amount=1):
        self.progress += amount

    def updateSubProgress(self, amount=1, reset=False):
        if reset: self.subProgress = 0
        else:
            self.subProgress += amount

    def clear(self):
        os.system("cls")
        os.system("clear")



class DataFormat:
    def __init__(self, resize):
        self.pLoader = ProgressLoader()
        self.resize = resize
        self.inputPath = "./xml/"
        self.directoryList, _ = self.getDirList(self.inputPath)

        num_cpus = psutil.cpu_count(logical=False)
        ray.init(num_cpus=num_cpus, log_to_driver=True)

    def getDirList(self, searchDir):
        for base, dirs, files in os.walk(searchDir):
            directoryList = []
            fileList = []
            for directory in dirs:
                directoryList.append(directory)
            for file in files:
                fileList.append(file)
            return directoryList, fileList

    @ray.remote
    def resizeImages(self, width, inputDir, outputDir):
        _, fileList = self.getDirList(inputDir)
        
        # self.pLoader.updateSubLength(len(fileList))
        # self.pLoader.updateSubProgress(reset=True)
        for i, fileName in enumerate(fileList[1:]):
            self.resizeImage(width, inputDir, outputDir, fileName)
        #     self.pLoader.updateSubProgress()
        #     self.pLoader.printIncrement(i, 4)
        # self.pLoader.updateProgress()

    def resizeImage(self, width, inputDir, outputDir, imgName):
        image = Image.open(inputDir + "/" + imgName)
        newImage = image.resize((width, width), Image.NEAREST)
        newImage.save(outputDir + "/" + imgName)

    def getRoot(self, dirName):
        xmlDir = glob.glob(self.inputPath + dirName + "/*.xml")[0]
        tree = ET.parse(xmlDir)
        root = tree.getroot()
        return root

    def getCategories(self, root):
        categoryList = []
        categoryDict = {}

        labels = root[1][0][12].findall('label')
        for i, label in enumerate(labels):
            category = {"supercategory": "none", "id": i+1, "name": label[0].text}
            categoryList.append(category)
            categoryDict[label[0].text] = i+1
        return (categoryList, categoryDict)



class YoloFormat(DataFormat):
    def __init__(self, resize, valid=20, test=5):
        super().__init__(resize)
        self.categoryDict = {}
        self.valid = valid
        self.test = test
        self.outputPath = "./yolo/"

        self.pLoader.updateLength(len(self.directoryList) + 1)
        self.createYolo()
        self.createYaml()
        self.resizeImages()
        self.shuffleData(self.outputPath + "train/images")

    def createYolo(self):
        outputList = ["train", "valid", "test"]
        for dirName in outputList:
            outputDir = self.outputPath + dirName
            os.mkdir(outputDir)
            os.mkdir(outputDir + "/images")
            os.mkdir(outputDir + "/labels")

        for dirName in self.directoryList:
            root = self.getRoot(dirName)
            _, subDict = self.getCategories(root)
            for key, value in subDict.items():
                self.categoryDict[key] = None
        for i, pair in enumerate(self.categoryDict.items()):
            self.categoryDict[pair[0]] = i

        progressText = "Creating Labels"
        self.pLoader.updateSubProgress(reset=True)
        self.pLoader.updateSubLength(len(self.directoryList))
        for i, dirName in enumerate(self.directoryList):
            root = self.getRoot(dirName)
            self.createLabels(root)
            self.pLoader.updateSubProgress()
            self.pLoader.printIncrement(i, progressText, 4)

        self.pLoader.updateProgress()
        self.pLoader.printWhole("Done")

    def shuffleData(self, dir):
        fileList = []
        count = 0
        for base, dirs, files in os.walk(dir):
            for file in files:
                count += 1
                fileList.append(file)

        validCount = int(count / 100.0 * self.valid)
        testCount = int(count / 100.0 * self.test)
        sampleList = sample(fileList, validCount+testCount)
        validList = sampleList[:validCount]
        testList = sampleList[validCount:]
        source = self.outputPath + "train/"

        dest = self.outputPath + "valid/"
        for imgName in validList:
            shutil.move(source+"images/"+imgName, dest+"images")
            labelName = imgName[:-4] + ".txt"
            shutil.move(source+"labels/"+labelName, dest+"labels")

        dest = self.outputPath + "test/"
        for imgName in testList:
            shutil.move(source+"images/"+imgName, dest+"images")
            labelName = imgName[:-4] + ".txt"
            shutil.move(source+"labels/"+labelName, dest+"labels")

    def createYaml(self):
        with open(self.outputPath + "data.yaml", "w", encoding="utf-8") as f:
            f.write("train: ../train/images" + "\n")
            f.write("val: ../valid/images" + "\n")
            f.write("test: ../test/images" + "\n\n")
            f.write("nc: " + str(len(self.categoryDict)) + "\n")
            f.write("names: " + "['" + "','".join(self.categoryDict.keys()) + "']")
            f.close()

    def resizeImages(self):
        refs = []
        for dirName in self.directoryList:
            refs.append(super().resizeImages.remote(
                    self, self.resize, self.inputPath + dirName, self.outputPath + "train/images"
                )
            )
        ray.get(refs)

    def createLabels(self, root):
        imageTags = root.findall("image")
        for i, imageTag in enumerate(imageTags):
            name = imageTag.attrib['name'][:-4]
            height = float(imageTag.attrib['height'])
            width = float(imageTag.attrib['width'])

            widthDiv, heightDiv = width/self.resize, height/self.resize
            outputDir = self.outputPath + "train/labels/"
            with open(outputDir + name + ".txt", "w") as f:
                for boxTag in imageTag.findall("box"):
                    attr = boxTag.attrib
                    label = attr["label"]
                    xtl, ytl, xbr, ybr = float(attr['xtl']), float(attr['ytl']), \
                        float(attr['xbr']), float(attr['ybr'])
                    xtl, ytl, xbr, ybr = xtl/widthDiv, ytl/heightDiv, xbr/widthDiv, ybr/heightDiv

                    category = self.categoryDict[label]
                    xCenter = ((xtl+xbr) / 2.0) / self.resize
                    yCenter = ((ytl+ybr) / 2.0) / self.resize
                    boxWidth = abs(xtl-xbr) / self.resize
                    boxHeight = abs(ytl-ybr) / self.resize

                    outputLine = "{} {} {} {} {}".format(category, xCenter, yCenter, boxWidth, boxHeight)
                    f.write(outputLine + "\n")



class CocoFormat(DataFormat):
    def __init__(self, resize):
        super().__init__(resize)

        self.ouputPath = "./coco/"

    def createCoco(self):
        self.pLoader.updateLength(len(self.directoryList))
        for dirName in self.directoryList:
            root = self.getRoot(dirName)
            info = self.getInfo(root)
            licenses = self.getLicenses()
            categories, categoryDict = self.getCategories(root)
            images = self.getImages(root)
            annotations = self.getAnnotations(root, categoryDict)

            coco = {
                "info": info,
                "licenses": licenses,
                "images": images,
                "annotations": annotations,
                "categories": categories,
                "segment_info": []
            }

            inputDir = self.inputPath + dirName
            outputDir = self.ouputPath + dirName

            os.mkdir(outputDir)
            with open(outputDir + "/" + dirName + ".json", "w", encoding="utf-8") as f:
                json.dump(coco, f, ensure_ascii=False, indent=4)

            self.resizeImages(inputDir, outputDir)

        self.pLoader.printWhole()

    def getInfo(self, root):
        task = root[1][0]

        description = ""
        url = task[13][0][3].text
        version = root[0].text
        year = 2019
        contributor = ""
        date_created = task[7].text

        return {
            "description": description, "url": url, "version": version, "year": year,
            "contributor": contributor, "date_created": date_created
        }

    def getLicenses(self):
        return {
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License"
        }

    def getImages(self, root):
        imageList = []

        for imageTag in root.findall("image"):
            attr = imageTag.attrib
            id = int(attr['id'])
            name = attr['name']
            height = self.resize
            width = self.resize

            image = {
                "license": 1,
                "file_name": name,
                "coco_url": "",
                "height": height,
                "width": width,
                "date_captured": "",
                "flickr_url": "",
                "id": id
            }
            imageList.append(image)

        return imageList

    def getAnnotations(self, root, categoryDict):
        annotationList = []

        for imageTag in root.findall("image"):
            imgAttr = imageTag.attrib
            for i, boxTag in enumerate(imageTag):
                attr = boxTag.attrib
                segmentation = []

                imgWidth, imgHeight = float(imgAttr['width']), float(imgAttr['height'])
                widthDiv, heightDiv = imgWidth / self.resize, imgHeight / self.resize
                xtl, ytl, xbr, ybr = float(attr['xtl']), float(attr['ytl']), \
                    float(attr['xbr']), float(attr['ybr'])
                xtl, ytl, xbr, ybr = xtl/widthDiv, ytl/heightDiv, xbr/widthDiv, ybr/heightDiv

                area = 0
                iscrowd = 0
                image_id = int(imgAttr['id'])
                bbox = [xtl, ytl, abs(xtl-xbr), abs(ytl-ybr)]
                category_id = categoryDict[attr['label']]
                id = int("{}{}{}".format(image_id, 0, i))

                annotation = {
                    "segmentation": segmentation,
                    "area": area,
                    "iscrowd": iscrowd,
                    "image_id": image_id,
                    "bbox": bbox,
                    "category_id": category_id,
                    "id": id
                }
                annotationList.append(annotation)
        
        return annotationList




if __name__ == '__main__':
    yolo = YoloFormat(1280)