# HAPT
   HarvardX: PH125.9x  Data Science: Capstone Smartphone-Based Recognition of Human Activities and Postural Transitions

## READ THIS BEFORE RUN THE CODE!!!            
                                                        
  1. This code is made for R version 4.1.2 (2021-11-01) Check your R version (type "R.version.string" in console) and update if needed;           
                                                        
  2. Check Program Control section in the code: it contains RETRAIN flag, if it set to TRUE, all models will be retrained and saved to folder "models". Running time will be ~10-14 hours, depends on computer;
  If RETRAIN set to False, then, models will not be retrained, but loaded from "models" folder, or, if not found, downloaded from GitHub. Total size is ~1GB;                           
                                                        
  3. Program checks if dataset is in the "data" folder and downloads it if it is not a case. Dataset size is 75.9MB;                 
                                                        
  4. Because program works with files in different folders, it is important to set correct working directory. It is done automatically, by using function: setwd(dirname(getActiveDocumentContext()$path)).   
  Please, keep cursor in the source window during code running to avoid wrong setting WD;
