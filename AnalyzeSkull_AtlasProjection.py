import os
import numpy as np
import subprocess


def AnalyzeSkull_AtlasProjection():
    programsFolder = "C:/Users/Austin Tapp/Documents/CraniosynostosisSlicer411Extn/Craniosynostosis/Executables"
    QuantifyMalformationsProgram = os.path.join(programsFolder, 'QuantifyDeformities.exe')
    resampleMeshProgram = os.path.join(programsFolder, 'UndersampleMesh.exe')
    referenceMeshFileName = 'AtlasConstrainedProjectionMesh_75.vtk'
    landmarksFileName = 'LandmarksSubject.fcsv'
    malformationsMeshFileName = 'Malformations_AtlasProjection_75.vtk' # new
    casesPath = NodeUtils.GetModelDataPath() + "/Tmp"
    cranialMeshFileName = 'CutBinarySubjectModel.vtk'

    # Looking in the database
    nCasesFound = 0
    for root, dirs, files in os.walk(casesPath):
        # It is a case folder
        if os.path.isdir(root):
            # If it has patient data
            if os.path.exists(os.path.join(root, referenceMeshFileName)):
                (path,folderName) = os.path.split(root)
                print('Processing: ' + root)
                nCasesFound = nCasesFound + 1
                landmarksPath = os.path.join(root, landmarksFileName)

                # Resampling mesh
                command = '\"' + resampleMeshProgram+'\" \"' +os.path.join(root, cranialMeshFileName)+'\" \"' + landmarksPath +'\" \"' + os.path.join(root, malformationsMeshFileName) + '\" -ignoreSutures'
                print('Running US mesh: ' + command)
                subprocess.call(command, shell=True)

                # Calculating malformations with external software
                mal_command = '\"' + QuantifyMalformationsProgram+'\" \"'+os.path.join(root, malformationsMeshFileName)+'\" \"' + os.path.join(root, referenceMeshFileName) +'\" \"' + os.path.join(root, malformationsMeshFileName) + '\"'
                print('Quantifying Malformations: ' + mal_command)
                subprocess.call(mal_command, shell=True)