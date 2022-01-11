from PIL import Image
from random import sample

import xml.etree.cElementTree as ET
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

        self.printProgress()

    def printProgress(self):
        self.clear()
        progress = (self.progress / self.length * 100.0) * (self.blockCount / 10.0)
        subProgress = (self.subProgress / self.subLength * 100.0) * (self.blockCount / 10.0)
        count = int(progress / 10.0)
        subCount = int(subProgress / 10.0)
        progressBar = "█" * count + "░" * (self.blockCount - count)
        subProgressBar = "█" * subCount + "░" * (self.blockCount - subCount)
        print(progressBar)
        print()
        print(subProgressBar)
        print("Processing {:.1f}%...".format(subProgress / (self.blockCount / 10.0)))

    def clear(self):
        os.system("cls")
        os.system("clear")



class DataFormat:
    def __init__(self, resize):
        self.pLoader = ProgressLoader()
        self.resize = resize
        self.inputPath = "./xml/"
        self.directoryList, _ = self.getDirList(self.inputPath)

    def getDirList(self, searchDir):
        for base, dirs, files in os.walk(searchDir):
            directoryList = []
            fileList = []
            for directory in dirs:
                directoryList.append(directory)
            for file in files:
                fileList.append(file)
            return directoryList, fileList

    def resizeImages(self, inputDir, outputDir):
        progressSpeed = 7
        _, fileList = self.getDirList(inputDir)
        subLength = len(fileList)
        self.pLoader.subLength = subLength
        self.pLoader.subProgress = 0
        for i, fileName in enumerate(fileList[1:]):
            self.resizeImage(inputDir, outputDir, fileName)
            self.pLoader.subProgress += 1
            if i % int(subLength / progressSpeed) == 0:
                self.pLoader.printProgress()
        self.pLoader.progress += 1

    def resizeImage(self, inputDir, outputDir, imgName):
        image = Image.open(inputDir + "/" + imgName)
        newImage = image.resize((640, 640), Image.NEAREST)
        newImage.save(outputDir + "/" + imgName)

    def getRoot(self, dirName):
        xmlDir = glob.glob(self.inputPath + dirName + "/*.xml")[0]
        tree = ET.parse(xmlDir)
        root = tree.getroot()
        return root

    def getCategories(self, root):
        categoryList = []
        categoryDict = {}
        for i, label in enumerate(root[1][0][12].findall('label')):
            category = {"supercategory": "none", "id": i+1, "name": label[0].text}
            categoryList.append(category)
            categoryDict[label[0].text] = i+1
        return (categoryList, categoryDict)



class YoloFormat(DataFormat):
    def __init__(self, resize, valid=20, test=10):
        super().__init__(resize)
        self.categoryDict = {}
        self.valid = valid
        self.test = test
        self.outputPath = "./yolo/"

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

        self.pLoader.length = len(self.directoryList)
        for dirName in self.directoryList:
            root = self.getRoot(dirName)
            _, subDict = self.getCategories(root)
            for key, value in subDict.items():
                self.categoryDict[key] = None
        for i, pair in enumerate(self.categoryDict.items()):
            self.categoryDict[pair[0]] = i

        for dirName in self.directoryList:
            root = self.getRoot(dirName)
            self.createLabels(root)

        self.pLoader.printProgress()

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
        for dirName in self.directoryList:
            super().resizeImages(self.inputPath + dirName, self.outputPath + "train/images")

    def createLabels(self, root):
        for imageTag in root.findall("image"):
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
        self.pLoader.length = len(self.directoryList)
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

        self.pLoader.printProgress()

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
    yolo = YoloFormat(640.0)