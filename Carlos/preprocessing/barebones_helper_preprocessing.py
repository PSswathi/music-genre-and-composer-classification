# Extra imports
import os
import shutil
import midipy
from midipy.midi_parser import parser_segments
from midipy.midi_parser import parser
from shutil import copytree, ignore_patterns
import zipfile
import pandas as pd
import sweetviz as sv

def folder_initializer(output_dir, extracted_features_dir, showviz_reports_dir, train_test_val_dir, chunked_output_dir, composers):
    #https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder
    #Initialize folders a priori

    output_directories = [output_dir, extracted_features_dir, showviz_reports_dir, train_test_val_dir, chunked_output_dir]
    dataset_types = ["train", "test", "val"]

    for directory in output_directories:
        for filename in os.listdir(directory):
            try:
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    for directory in output_directories:
        if(directory == output_dir):
            file = 'placeholder.txt'
            placeholder_path = os.path.join(directory, file)
            create_file(placeholder_path)

            for composer in composers:
                try:
                    composer_path = os.path.join(directory, composer)
                    if not os.path.exists(composer_path):
                        os.makedirs(composer_path)
                except Exception as e:
                    print('Failed to create subfolder %s. Reason: %s' % (composer_path, e))

            os.remove(placeholder_path)

        elif(directory == chunked_output_dir):
            file = 'placeholder.txt'
            placeholder_path = os.path.join(directory, file)
            create_file(placeholder_path)

            for dataset_type in dataset_types:
                try:
                    dataset_type_path = os.path.join(directory, dataset_type)
                    if not os.path.exists(dataset_type_path):
                        os.makedirs(dataset_type_path)
                except Exception as e:
                    print('Failed to create subfolder %s. Reason: %s' % (dataset_type_path, e))

            os.remove(placeholder_path)

        
    print(f'Succesfully initialized {output_directories}.')

def create_file(path_file):
    # Creating a file at specified location
    file = open(path_file, "w")
    file.close()

def selected_composer_extractor(input_dir, output_dir, composers):

    for foldername in os.listdir(input_dir):
        try:
            #Find files from those four specific composers
            if(foldername in composers):
                dest_path = str(output_dir + foldername + "/")
                shutil.copytree(str(input_dir + foldername), dest_path, dirs_exist_ok=True, ignore=ignore_patterns('.mid'))
                #recursive_mid_finder(dest_path, dest_path)
            else:
                next
        except Exception as e:
            print('Failed to copy the files of the composers %s. Reason: %s' % (foldername, e))

    print(f'Succesfully selected composers {composers} into {output_dir}. This is a test run')

def recursive_mid_finder(dest_path, file_name):
    for file in os.listdir(file_name):
        if((".mid" in str(file)) or (".MID" in str(file))):
            next
        elif(".zip" in str(file)):
            file_name = str(file_name + file)
            with zipfile.ZipFile(file_name, 'r') as zip_ref:
                zip_ref.extractall(dest_path)
            os.remove(file_name)
            file_name = str(file_name.replace(file,""))
        else:
            if(os.path.isdir(str(file_name + file + "/"))):
                file_name = str(file_name + file + "/")
                recursive_mid_finder(dest_path, file_name)
                shutil.copytree(file_name, dest_path, dirs_exist_ok=True)
                shutil.rmtree(file_name)
                file_name = str(file_name.replace(str(file + "/"),""))
            else:
                file_name = str(file_name + file)
                os.remove(file_name)
                file_name = str(file_name.replace(file,""))

def midi_feature_composer_extractor(input_dir, output_dir, composers):

    for composer in composers:

        iter_input_dir = os.path.join(input_dir, composer)
        iter_output_dir = os.path.join(output_dir, composer)

        print(f'Extracting midi features for {composer}.')

        try:
            # Default usage with optional overrides:
            df_default = parser(
                source=iter_input_dir,
                metrics=['Total_Counts', 'Avg_Velocity'],
                output_format='csv',
                save_path=composer
            )
            
            iter_input_dir = str(iter_input_dir.replace(str("/" + composer),""))
            shutil.move(str(composer + ".csv"), output_dir)
            
        except Exception as e:
            print('Failed to turn the Midi files for %s. Reason: %s' % (composer, e))
    print(f'Succesfully extracted Composer-level midi features to csvs.')

def midi_feature_composer_aggregator(extracted_features_dir):

    df_csv_append = pd.DataFrame()

    try:
        for filename in os.listdir(extracted_features_dir):
            file_path = os.path.join(extracted_features_dir, filename)
            df = pd.read_csv(file_path)
            df['composer'] = str(filename.replace(".csv",""))
            df_csv_append = pd.concat([df_csv_append, df], ignore_index=True)

        df_csv_append['composer'] = df_csv_append['composer'].astype('category')
        return df_csv_append

    except Exception as e:
        print('Failed to aggregate the Midi files to a DataFrame. Reason: %s' % (e))
    print(f'Succesfully aggregated the csvs to a DataFrame folder.')

def showviz_report_creator(showviz_reports_dir, dataframe, popup=False):
    try:
        for composer in dataframe['composer'].cat.categories:

            df_composer = dataframe
            df_composer.loc[df_composer["composer"] == composer, "isComposer"] = 1
            df_composer.loc[df_composer["composer"] != composer, "isComposer"] = 0
            #df_composer = df_composer.drop('composer', axis=1)
            df_composer["isComposer"].astype(bool)

            #https://stackoverflow.com/questions/76501976/sweetviz-installed-but-wont-run-module-numpy-has-no-attribute-warnings
            composer_path = showviz_reports_dir + 'sweetviz_eda_' + composer + '.html'
            sweet_report = sv.analyze(df_composer, target_feat="isComposer")
            sweet_report.show_html(composer_path, open_browser=popup)

        sweet_report_path = showviz_reports_dir + 'sweetviz_all_eda.html'
        sweet_report = sv.analyze(dataframe)
        sweet_report.show_html(sweet_report_path, open_browser=popup)

    except Exception as e:
        print('Failed to aggregate the Midi files to a %s DataFrame. Reason: %s' % (composer, e))
    print(f'Succesfully aggregated the csvs to a DataFrame folder.')

def midi_feature_segment_extractor(input_dir, output_dir, composers):

    for composer in composers:

        iter_input_dir = os.path.join(input_dir, composer)
        iter_output_dir = os.path.join(output_dir, composer)

        print(f"Chunking {composer}...")
        print(f"Chunking from {iter_input_dir}...")

        try:

            # Segment-wise analysis:
            df_segments = parser_segments(
                source=iter_input_dir,
                num_segments=3,
                metrics=['Total_Counts', 'Avg_Velocity'],
                output_format='csv',
                save_path=composer
            )

            iter_input_dir = str(iter_input_dir.replace(str("/" + composer),""))
            shutil.move(str(composer + ".csv"), output_dir)
            
        except Exception as e:
            print('Failed to chunk the Midi files for %s. Reason: %s' % (composer, e))
    print(f'Succesfully extracted midi features to csvs.')

def intermediary_folder_handler(train_test_val_dir, chunked_out_dir, composers):

    #might not hande no validation datasets well
    dataset_types = ["train", "test", "val"]

    # first up a subfolder all available
    for dataset_type in dataset_types:
        base_path = os.path.join(train_test_val_dir, dataset_type)
        destination_path = os.path.join(chunked_out_dir, dataset_type)
        for composer in composers:
            composer_path = os.path.join(base_path, composer)
            for file in os.listdir(composer_path):
                src_path = os.path.join(composer_path, file)
                dst_path = os.path.join(destination_path, file)
                os.rename(src_path, dst_path)

    print(f'Succesfully transfered split files to chunked folder.')