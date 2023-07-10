import os
import logging
from lxml import etree
import sys
import random
from datetime import datetime, timedelta

#############################################################
def replaceDate(in_str):
	#Expects date in order of dd mm yyyy
	dd = ''
	mm =''
	yy = ''
	dd_reset = False
	mm_reset = True
	yy_reset = True
	format_ch = ''
	for ch in in_str:
		if ch.isdigit():
			if dd_reset == False:
				dd = dd + ch
			if mm_reset == False:
				mm = mm + ch
			if yy_reset == False:
				yy = yy + ch		
		else:
			format_ch = ch
			dd_reset = True
			if mm_reset == True:
				mm_reset = False
			else:
				mm_reset = True
				yy_reset = False
	
	date_format = ''
	date_string = dd + format_ch + mm + format_ch + yy

	if len(yy) == 2:
		date_format = '%d' + format_ch + '%m' + format_ch + '%y'
	else:
		date_format = '%d' + format_ch + '%m' + format_ch + '%Y'

	date_dt = datetime.strptime(date_string, date_format)
	date_dt = date_dt + timedelta(days=2000)
	return datetime.strftime(date_dt, date_format)

#############################################################
def replaceDigit(in_str):
	result = []
	for ch in in_str:
		if ch.isdigit():
			result.append(getRandomDigit(ch))
		else:
			result.append(ch)
	return ''.join(result)

#############################################################
def replaceChar(in_str):
	result = []
	for ch in in_str:
		if ch.isalpha():
			result.append(getRandomChar(ch.lower()))
		else:
			result.append(ch)	
	return ''.join(result)

#############################################################
def getRandomChar(exceptThis):
	ascii_lowercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
	return getRandom(list(ascii_lowercase), exceptThis)

#############################################################
def getRandomDigit(exceptThis):
	nums = '0123456789'
	return getRandom(list(nums), exceptThis)

#############################################################
def getRandom(lst, exceptThis):
	if exceptThis != None and exceptThis in lst:
		lst.remove(exceptThis)
	return random.choice(lst)

#############################################################
def getList(file_name):
	f_list = []
	a_file = open("dictionaries/" + file_name, "r")
	for line in a_file: 
		f_list.append(line.strip())
	return f_list	

#############################################################
def getReplacementName(replacement, token, ch):
	random_name = None
	try:
		female_first_names.index(token)
		random_name = getRandom(female_first_names, token)
	except ValueError:
		pass
		
	try:		
		male_first_names.index(token)
		random_name = getRandom(male_first_names, token)
	except ValueError:
		pass
			
	try:
		last_names.index(token)
		random_name = getRandom(last_names, token)
	except ValueError:
		pass		
		
	if random_name != None:	
		replacement = replacement + random_name + ch
	else:
		replacement = replacement + token + ch
	return replacement
	
#############################################################		
if __name__ == '__main__':
	from_dir = os.path.abspath('') + '/input/'
	to_dir = os.path.abspath('') + '/output/'
			
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	
	cities = getList('CITY.txt');
	countries = getList('COUNTRY.txt');
	hospitals = getList('HOSPITAL.txt');
	organizations = getList('ORGANIZATION.txt');
	states = getList('STATE.txt');
	zips = getList('ZIP.txt');
	female_first_names = getList('FEMALE_FIRST_NAMES.txt');
	male_first_names = getList('MALE_FIRST_NAMES.txt');
	last_names = getList('LAST_NAMES.txt');
	
	logging.info('Start..')
	if os.path.isdir(from_dir):
		for fname in os.listdir(from_dir):
			if fname.endswith("xml"):
				try:
					doc = etree.parse(from_dir + fname)
					textEl = doc.find('TEXT')
					patient_text = ''
					if textEl is not None:
						try:
							patient_text = textEl.text.decode('utf-8')
							patient_text = patient_text.encode('utf8');
						except UnicodeDecodeError:
							logging.error('{} : is not valid UTF-8 encoded file.'.format(fname))
					else:
						logging.error('{} : Text element not found'.format(fname));
						sys.exit(0)
					tagEl = doc.find('TAGS')
					replace_dict = {}
					replace_dict['CITY'] = []
					replace_dict['COUNTRY'] = []
					replace_dict['HOSPITAL'] = []
					replace_dict['NAME'] = []
					replace_dict['ORGANIZATION'] = []
					replace_dict['STATE'] = []
					replace_dict['STATE'] = []
					if tagEl is not None:
						for child in tagEl.iterchildren("*"):
							tagName = child.tag
							tagAttr = child.attrib
							text = tagAttr['text'] 
							tag_type = tagAttr['TYPE']
							start = int(tagAttr['start'])
							end = int(tagAttr['end'])
							if tagName == 'ID' or tagName == 'AGE' or tagName == 'CONTACT':
								replacement = replaceDigit(text)
								a_list = list(patient_text)
								a_list[start:end] = replacement
								patient_text = ''.join(a_list)
							elif tagName == 'DATE':
								replacement = replaceDate(text)
								a_list = list(patient_text)
								a_list[start:end] = replacement
								patient_text = ''.join(a_list)
							elif tagName == 'LOCATION' or tagName == 'NAME' or tagName == 'PROFESSION':	
								if tag_type == 'CITY':
									replace_dict['CITY'].append(text) 
								elif tag_type == 'COUNTRY':
									replace_dict['COUNTRY'].append(text)
								elif tag_type == 'PATIENT' or tag_type == 'DOCTOR':
									replace_dict['NAME'].append(text)
								elif tag_type == 'HOSPITAL':
									replace_dict['HOSPITAL'].append(text)
								elif tag_type == 'ORGANIZATION':
									replace_dict['ORGANIZATION'].append(text)
								elif tag_type == 'STATE':
									replace_dict['STATE'].append(text)
								elif tag_type == 'ZIP':
									replace_dict['STATE'].append(text)
								else:	
									replacement = replaceChar(text)
									a_list = list(patient_text)
									a_list[start:end] = replacement
									patient_text = ''.join(a_list)	
							else:
								replacement = replaceChar(text)
								a_list = list(patient_text)
								a_list[start:end] = replacement
								patient_text = ''.join(a_list)			
					
					for key, values in replace_dict.iteritems():
						if key == 'CITY':
							for item in values:
								random_city = getRandom(cities, item)
								patient_text = patient_text.replace(item, random_city)
						elif key == 'COUNTRY':
							for item in values:
								random_country = getRandom(countries, item)
								patient_text = patient_text.replace(item, random_country)
						elif key == 'NAME':
							for item in values:
								replacement = ''
								token = ''
								for ch in item:
									if ch.isalpha():
										token = token + ch
									else:
										replacement = getReplacementName(replacement, token, ch)
										token = ''
								replacement = getReplacementName(replacement, token, '')
								if replacement == item:
									replacement = replaceChar(replacement);
								patient_text = patient_text.replace(item, replacement)
						elif key == 'HOSPITAL':
							for item in values:
								random_hospital = getRandom(hospitals, item)
								patient_text = patient_text.replace(item, random_hospital)
						elif key == 'ORGANIZATION':
							for item in values:
								random_org = getRandom(organizations, item)
								patient_text = patient_text.replace(item, random_org)
						elif key == 'STATE':
							for item in values:
								random_state = getRandom(states, item)
								patient_text = patient_text.replace(item, random_state)
						elif key == 'ZIP':
							for item in values:
								random_zip = getRandom(zips, item)
								patient_text = patient_text.replace(item, random_zip)
						
					fileName = os.path.splitext(fname)[0]
					fw = open(os.path.join(to_dir, fileName + '.txt'), "wb+")
					fw.write(patient_text.encode('utf-8'))
					fw.close()
					logging.info('Done Parsing file:' + fname)			
				except etree.XMLSyntaxError:
					logging.error('{} : is not a valid xml file'.format(fname))
	logging.info('Done...')				
