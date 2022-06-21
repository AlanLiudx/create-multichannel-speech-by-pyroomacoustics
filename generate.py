import pyroomacoustics as pra
import soundfile as sf
import librosa
import numpy as np
import os
import random
import reset_simulator_state
import json
import tqdm
from tqdm import tqdm
from audiomentations.core.utils import find_audio_files_in_paths,find_audio_files,calculate_desired_noise_rms,convert_decibels_to_amplitude_ratio,calculate_rms
#We generate our roomacoustics with pyroomacoustics. This allows us use simulated distance degrade instead of a given SNR

#attatched global methods
def Normalize_wave(samples):
    max_abs=np.amax(np.abs(samples))
    if max_abs > 0:
        normalized_samples = samples / max_abs
    else:
        normalized_samples = samples
    return normalized_samples

#basic config of our dataset:
nums_of_all=40000
nums_of_train=28000
nums_of_valid=8000
nums_of_test=4000

all_dir_prefix="/data/liudx/farfield_dataset_with_raytracing"
if not os.path.isdir(all_dir_prefix):
    os.mkdir(all_dir_prefix)
train_dir=os.path.join(all_dir_prefix,"train")
if not os.path.isdir(train_dir):
    os.mkdir(train_dir)
valid_dir=os.path.join(all_dir_prefix,"valid")
if not os.path.isdir(valid_dir):
    os.mkdir(valid_dir)
test_dir=os.path.join(all_dir_prefix,"test")
if not os.path.isdir(test_dir):
    os.mkdir(test_dir)
#the size of samples of every speech, 64000 samples
trunk_size=64000
# ##create a train bar
pbar=tqdm(range(4000))

#corpus direction
speech_corpus_dir=["/data/share/datasets/librispeech_360/train-clean-360/","/data/share/datasets/aishell_1/data_aishell/wav/"]

noise_corpus_dir=["/data/share/datasets/DEMAND/","/data/share/datasets/Noise/noise_200/"]

#should we add interfererence?
interference_corpus_dir="/data/share/datasets/musan/speech/"


# get file list. This needs audiomentations library
speech_file_list=find_audio_files_in_paths(speech_corpus_dir)
noise_file_list=find_audio_files_in_paths(noise_corpus_dir)
interference_file_list=find_audio_files_in_paths(interference_corpus_dir)

#initialize the relative position of microphones.
microphone_basic_coordinate = np.c_[[0,0,0]] #the coordinate should be changed
microphone_basic_coordinate[0] = 2.0
microphone_basic_coordinate[1] = 2.0
microphone_basic_coordinate[2] = 2.0
#microphone array radius:8cm. This is different from that of ray_tracing radius
microphone_array_radius=0.08

mic_locs_relative = np.c_[
    [microphone_array_radius*np.cos(0*0.25*np.pi), microphone_array_radius*np.sin(0*0.25*np.pi), 0],  # mic 1
    [microphone_array_radius*np.cos(1*0.25*np.pi), microphone_array_radius*np.sin(1*0.25*np.pi), 0],  # mic 2
    [microphone_array_radius*np.cos(2*0.25*np.pi), microphone_array_radius*np.sin(2*0.25*np.pi), 0],  # mic 3
    [microphone_array_radius*np.cos(3*0.25*np.pi), microphone_array_radius*np.sin(3*0.25*np.pi), 0],  # mic 4
    [microphone_array_radius*np.cos(4*0.25*np.pi), microphone_array_radius*np.sin(4*0.25*np.pi), 0],  # mic 5
    [microphone_array_radius*np.cos(5*0.25*np.pi), microphone_array_radius*np.sin(5*0.25*np.pi), 0],  # mic 6
    [microphone_array_radius*np.cos(6*0.25*np.pi), microphone_array_radius*np.sin(6*0.25*np.pi), 0],  # mic 7
    [microphone_array_radius*np.cos(7*0.25*np.pi), microphone_array_radius*np.sin(7*0.25*np.pi), 0],  # mic 8
]
mic_locs = mic_locs_relative + microphone_basic_coordinate
# mic_locs = mic_locs_relative + microphone_basic_coordinate

#no direction difference should be smaller than 15 degree i.e. 15/360*2*pi=np.pi/12
minimum_direction_difference=np.pi/12

#reverb settings
reverb_config=['anechoic','medium_reverb','low_reverb']

#iteration:
for index in pbar:
    #initialize information json
    info_forjson={}
    #initialize noise source nums
    #mind: the amplitude before mixing should be divided by noise_source_nums
    noise_source_nums=np.random.randint(1,7)
    info_forjson["noise_source_nums"]=noise_source_nums
    ##if we need speech-like interference,0 means no interference
    if_interference=np.random.randint(0,2)
    info_forjson["if_interference"]=if_interference
    #randomly choose speech name
    speech_index=np.random.randint(0,len(speech_file_list))
    speech_name=speech_file_list[speech_index]
    info_forjson["speech_name"]=str(speech_name)
    
    #randomly choose noise name
    noise_name_set=[]
    for num_noise in range(noise_source_nums):
        if if_interference and num_noise==0:
            noise_index=np.random.randint(0,len(interference_file_list))
            noise_name=interference_file_list[noise_index]
        else:
            noise_index=np.random.randint(0,len(noise_file_list))
            noise_name=noise_file_list[noise_index]
        noise_name_set.append(noise_name)
   
    random.shuffle(noise_name_set)
    noise_name_set_forjson=noise_name_set.copy()
    for i in range(len(noise_name_set_forjson)):
        noise_name_set_forjson[i]=str(noise_name_set_forjson[i])
    info_forjson["noise_name_set"]=noise_name_set_forjson
    #randomly choose speech direction
    source_direction=np.random.uniform(0,2*np.pi)
    info_forjson["source_direction"]=source_direction
    #randomly choose noise direction
    noise_direction_list=[]
    noise_first_direction=np.random.uniform(source_direction+minimum_direction_difference,2*np.pi+source_direction-minimum_direction_difference)
    info_forjson["noise_first_direction"]=noise_first_direction
    while(noise_first_direction >= 2*np.pi):
        noise_first_direction -= 2*np.pi
    interval_noise=2*np.pi/noise_source_nums
    for num_noise in range(noise_source_nums):
        noise_direction_list.append(noise_first_direction + num_noise * interval_noise)
    
    #randomly choose speech distance, from 4m to 25m
    source_distance=np.random.uniform(4.0,25.0)
    info_forjson["source_distance"]=source_distance
    #randomly choose noise distance, from 4m to 10m
    noise_distance=np.random.uniform(4.0,10.0)
    info_forjson["noise_distance"]=noise_distance
    #We use hybrid ISM method: max_order=3 and later reverb is modeled by ray_tracing method. We have to generate e_absorption
    reverb_situation=random.choice(reverb_config)
    info_forjson["reverb_situation"]=reverb_situation
    #randomly set e_absorption
    if(reverb_situation=='anechoic'):
        e_absorption=1.0
    elif (reverb_situation=='low_reverb'):
        e_absorption=np.random.uniform(0.7,1.0)
    elif (reverb_situation=='medium_reverb'):
        e_absorption=np.random.uniform(0.4,0.7)
    
    info_forjson["e_absorption"] = e_absorption
    
    #set room dim
    max_dist=np.max([noise_distance,source_distance])
    room_dim=[2*(2+max_dist),2*(2+max_dist),25]
    #first create anechoic reference
    room = pra.ShoeBox(
    room_dim, fs=16000, materials=pra.Material(e_absorption), max_order=0
)
    #set basic coordinate
    microphone_basic_coordinate[0] = 2.0+max_dist
    microphone_basic_coordinate[1] = 2.0+max_dist
    microphone_basic_coordinate[2] = 1.5
    mic_locs = mic_locs_relative + microphone_basic_coordinate
    #add microphone
    room.add_microphone_array(mic_locs)
    
    #file name
    if index<nums_of_train:
        index_in_section=index
        output_path_prefix=os.path.join(train_dir,"sample"+str(index_in_section))
        if not os.path.isdir(output_path_prefix):
            os.mkdir(output_path_prefix)
    
    elif index>=nums_of_train and index<(nums_of_valid + nums_of_train):
        index_in_section=index - nums_of_train
        output_path_prefix=os.path.join(valid_dir,"sample"+str(index_in_section))
        if not os.path.isdir(output_path_prefix):
            os.mkdir(output_path_prefix)
    
    elif index>=(nums_of_valid + nums_of_train) and index<(nums_of_valid + nums_of_train + nums_of_test):
        index_in_section=index - nums_of_train -nums_of_valid
        output_path_prefix=os.path.join(test_dir,"sample"+str(index_in_section))
        if not os.path.isdir(output_path_prefix):
            os.mkdir(output_path_prefix)
    
    #read speech
    speech_input , sr = sf.read(speech_name)
    if (len(speech_input)-trunk_size>0):
        section_point  =np.random.randint(0,len(speech_input)-trunk_size)   
        speech_input = speech_input[section_point:section_point+trunk_size].copy()
    else:
        speech_input=np.pad(speech_input,(0,trunk_size-len(speech_input)))
        
    speech_input = Normalize_wave(speech_input)
    
    #add source
    source_coord=[microphone_basic_coordinate[0]+ source_distance * np.cos(source_direction) , microphone_basic_coordinate[1] + source_distance * np.sin(source_direction) , microphone_basic_coordinate[2]]
    room.add_source(source_coord, signal=speech_input, delay=0.0)
    #add all noise sources
    for num_noise in range(noise_source_nums):
        noise_input , sr = sf.read(noise_name_set[num_noise])
        if (len(noise_input)-trunk_size>0):
            section_point  =np.random.randint(0,len(noise_input)-trunk_size)
            noise_input = noise_input[section_point:section_point+trunk_size].copy()
        else:
            noise_input=np.pad(noise_input,(0,trunk_size-len(noise_input)))
        noise_input = Normalize_wave(noise_input)
        # noise_input = noise_input / noise_source_nums #normalize the energy that we shouldn't let noise energy increase with num of noise sources
        #location
        noise_coord = [microphone_basic_coordinate[0]+ noise_distance * np.cos(noise_direction_list[num_noise]) , microphone_basic_coordinate[1] + noise_distance * np.sin(noise_direction_list[num_noise]) , microphone_basic_coordinate[2]]
        room.add_source(noise_coord, signal=noise_input, delay=0.0)
    
    # room.set_air_absorption()
    #zero order
    premix=room.simulate(recompute_rir=True,return_premix=True)
    #premix_(n_sources,n_mics,n_samples)
    #save anechoic speech ref and noise ref
    speech_ref_ane=premix[0,:,:].copy()
    noise_ane=np.mean(premix[1:,:,:].copy(),0)
    speech_ref_ane=speech_ref_ane.transpose()
    speech_ref_ane = speech_ref_ane[0:trunk_size,:]
    output_dir  =os.path.join(output_path_prefix,"speech.wav")
    sf.write(output_dir,speech_ref_ane,samplerate=16000)
    noise_ane=noise_ane.transpose()
    noise_ane = noise_ane[0:trunk_size,:]
    output_dir  =os.path.join(output_path_prefix,"noise.wav")
    sf.write(output_dir,noise_ane,samplerate=16000)
    

    #reset simulator state. Note that only ism_needed will be set True, and rt_needed as False
    room.reset_simulator_state()
    #change max order
    room.max_order=3
    premix=room.simulate(recompute_rir=True,return_premix=True)
    #save early reverb
    speech_ref_early=premix[0,:,:].copy()
    noise_early=np.mean(premix[1:,:,:].copy(),0)
    speech_ref_early=speech_ref_early.transpose()
    speech_ref_early = speech_ref_early[0:trunk_size,:]
    output_dir  =os.path.join(output_path_prefix,"speech_early.wav")
    sf.write(output_dir,speech_ref_early,samplerate=16000)
    noise_early=noise_early.transpose()
    noise_early = noise_early[0:trunk_size,:]
    output_dir  =os.path.join(output_path_prefix,"noise_early.wav")
    sf.write(output_dir,noise_early,samplerate=16000)
    
    #reset simulator state. Note that only ism_needed will be set True, and rt_needed as False
    room.reset_simulator_state()
    #set ray_tracing for further noisy mixture model, as well as air absorption 
    room.set_ray_tracing()
    room.set_air_absorption()
    #simulate mixture
    premix=room.simulate(recompute_rir=True,return_premix=True)
    #save early reverb
    speech_ref_late=premix[0,:,:].copy()
    noise_late=np.mean(premix[1:,:,:].copy(),0)
    speech_ref_late=speech_ref_late.transpose()
    speech_ref_late = speech_ref_late[0:trunk_size,:]
    noise_late=noise_late.transpose()
    noise_late = noise_late[0:trunk_size,:]
    output_dir  =os.path.join(output_path_prefix,"mixture.wav")
    mixture=speech_ref_late.copy()+noise_late.copy()
    sf.write(output_dir,mixture,samplerate=16000)

    
    #we finished writing waves, then for json
    json_output_path=os.path.join(output_path_prefix,"info.json")
        #dump json. We need indent
    info_json_data=json.dumps(info_forjson,indent=1)
    with open(json_output_path,"w",newline='\n') as JSON_FILE:
        JSON_FILE.write(info_json_data)
        
print("work complete")
print("the output directory is:",all_dir_prefix)
    

