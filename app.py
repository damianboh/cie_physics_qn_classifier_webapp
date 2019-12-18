import PyPDF2
from PyPDF2 import PdfFileWriter,PdfFileReader
from PyPDF2 import PdfFileMerger
import os
import pandas as pd
import pickle
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import lxml.html
from lxml import etree
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import HTMLConverter,TextConverter,XMLConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import io
import re
import glob
from zipfile import ZipFile
from jinja2 import Environment, FileSystemLoader, select_autoescape
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, send_file
from werkzeug import secure_filename

# Initialize the Flask application
app = Flask(__name__, template_folder='templates')

# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'tmp/'
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['pdf'])

output_folder = "tmp/sorted_output"


env = Environment(
    loader=FileSystemLoader('templates'),
    autoescape=select_autoescape(['html', 'xml'])
)
template = env.get_template('sorted.html')


with open(f'model/classifier.pkl', 'rb') as f:
   classifier = pickle.load(f)

with open(f'model/tfidf_vectorizer.pkl', 'rb') as f:
   tfidf_vectorizer = pickle.load(f)

def predict_topic(text):
    result = classifier.predict(tfidf_vectorizer.transform([text]))
    return(result[0])

def convert(case,fname, pages=None):
    if not pages: pagenums = set();
    else:         pagenums = set(pages);      
    manager = PDFResourceManager() 
    codec = 'utf-8'
    caching = True

    if case == 'text' :
        output = io.StringIO()
        converter = TextConverter(manager, output, codec=codec, laparams=LAParams())     
    if case == 'HTML' :
        output = io.BytesIO()
        converter = HTMLConverter(manager, output, codec=codec, laparams=LAParams())

    interpreter = PDFPageInterpreter(manager, converter)   
    infile = open(fname, 'rb')

    for page in PDFPage.get_pages(infile, pagenums,caching=caching, check_extractable=True):
        interpreter.process_page(page)

    convertedPDF = output.getvalue()  

    infile.close(); converter.close(); output.close()
    return convertedPDF

def pdf_to_html(pdf_file_name):
    filePDF = pdf_file_name
    fileHTML = 'test.html'
    fileTXT  = 'test.txt'     # output
    case = "HTML"

    if case == 'HTML' :
        convertedPDF = convert('HTML', filePDF)
        fileConverted = open(fileHTML, "wb")
    if case == 'text' :
        convertedPDF = convert('text', filePDF)
        fileConverted = open(fileTXT, "w")

    fileConverted.write(convertedPDF)
    fileConverted.close()
    html_string = convertedPDF.decode("utf-8")
    return html_string

def find_question_in_html(html_string):
    
    question_numbers = re.findall(r"""left:[0-5][0-9]px; top:\d+px; width:\d+px; height:\d+px;"><span style="font-family: b'(?:.*)-BoldM?T?'; font-size:1[0-9]px">\d+""", html_string)

    print("Number of questions: " + str(len(question_numbers)))
       
    question_numbers_corrected = []

    for i, question in enumerate(question_numbers):
        if "+" in question:
            plus_position = question.find("+")
            question_numbers_corrected.append(question[:plus_position] + "\\" + question[plus_position:])
        else:
            question_numbers_corrected.append(question)
            
    return question_numbers_corrected
	
def get_question_texts(question_numbers_corrected, html_string):
    question_texts = []

    for i, question in enumerate(question_numbers_corrected):
        if i == (len(question_numbers_corrected) - 1): # if it is the last question
            question_texts.append(re.findall(question_numbers_corrected[i] + "[\s\S]*", html_string))
        else:
            question_texts.append(re.findall(question_numbers_corrected[i] + "[\s\S]*" + question_numbers_corrected[i+1] , html_string))

    parser = etree.HTMLParser()
    
    question_text_notags = []

    for question_text in question_texts:
        tree = etree.fromstring(question_text[0], parser=parser)
        notags = etree.tostring(tree, encoding='utf8', method='text')
        question_text_notags.append(notags[46:])
        
    return(question_text_notags)
	
def get_pages(html_string):
    pages_text = re.findall(r"Page \d+", html_string)
    return(pages_text) 
    
	
def get_pages_of_questions(pages_text, question_numbers_corrected, html_string):
    question_page_dict = dict.fromkeys(range(1, len(question_numbers_corrected) + 1)) 
    df_page = pd.DataFrame()
    for p, page_text in enumerate(pages_text): 
        
        for q, question_number in enumerate(question_numbers_corrected):
                        
            if p == 0: # take away cover page (do nothing), somehow all questions are found in cover page
                continue
                
            elif p == (len(pages_text) - 1): # if it is the last page
                if(re.findall(pages_text[p] + "[\s\S]*" + question_number + "[\s\S]*", html_string)):
                    #print("Found question " + str(q+1) + " in page " + str(p+1))
                    question_page_dict[q+1] = [p+1]
                    
            else:
                if(re.findall(pages_text[p] + "[\s\S]*" + question_number + "[\s\S]*" + pages_text[p+1] , html_string)):
                    #print("Found question " + str(q+1) + " in page " + str(p+1))
                    question_page_dict[q+1] = [p+1]
    
    diff = 0
    
    #df_page = pd.DataFrame.from_dict(question_page_dict, orient = 'index')
    #df_page.columns = ['start page']
    #df_page.index.name = 'question'
    
    for key in (range(1, len(question_numbers_corrected) + 1)):
        if key == (len(question_numbers_corrected)):  # if it is the last question
            diff = len(pages_text) - question_page_dict[key][0]
        else:
            diff = question_page_dict[key+1][0] - question_page_dict[key][0]
            
            for number in range(1, diff):
                question_page_dict[key].append(question_page_dict[key][0] + number)
       
    return(question_page_dict)
    
def get_question_marks(question_text):
    marks = re.findall(r"\[(\d+)\]", question_text.decode("utf-8"))
    total_marks = 0
    for mark in marks:
        total_marks += int(mark)
        
    return total_marks

def add_question_to_pdf(readfile, page_list, writefile):    
       
    reader=PdfFileReader(open(readfile, 'rb'))
    
    
    if not os.path.exists(writefile + '.pdf'):
        outputStream = open(writefile + '.pdf', "wb")
        writer = PdfFileWriter()
    
        
        for page_no in page_list:
            writer.addPage(reader.getPage(page_no-1)) # -1 as page 1 is 0, page 2 is 1 etc.
        
        # write to file
        writer.write(outputStream)
        outputStream.close()
    
    else:
        outputStream = open(writefile + '_temporary.pdf', "wb")
        writer = PdfFileWriter()
        merger = PdfFileMerger()
        
        for page_no in page_list:
            writer.addPage(reader.getPage(page_no-1)) # -1 as page 1 is 0, page 2 is 1 etc.
        
        # write to file
        writer.write(outputStream)
        outputStream.close()
        
        merger.append(PdfFileReader(writefile + '.pdf'))
        merger.append(PdfFileReader(writefile + '_temporary.pdf'))

        
        merger.write(writefile + '.pdf')
        merger.close()

def predict_by_question(file_name, output_folder):
    print(file_name)
    try:
        html_string = pdf_to_html(file_name)
        question_numbers_corrected = find_question_in_html(html_string)
        pages_text = get_pages(html_string)
        question_text_notags = get_question_texts(question_numbers_corrected, html_string)
        df_predict_question = pd.DataFrame()
        
        for i, question in enumerate(question_text_notags):

            df2 = pd.DataFrame({"paper": [os.path.basename(file_name)], 
                                "question": [i+1], 
                                "predicted_topic": [predict_topic(question)], 
                                "predicted_marks":get_question_marks(question),
                                "start_page": 0})
                      
            df_predict_question = df_predict_question.append(df2)

        total_marks = df_predict_question['predicted_marks'].sum()
        print("Total marks: " + str(total_marks))
        df_predict_question = df_predict_question.set_index(['paper','question'])
        
        question_page_dict = get_pages_of_questions(pages_text, question_numbers_corrected, html_string)
        
        for i in range(len(df_predict_question)):
            df_predict_question.iloc[i,2] = question_page_dict[i+1][0] # remember the i + 1
        
        output_folder = output_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        page_list = []
        
        for i in range(len(df_predict_question)):
            page_list = question_page_dict[i+1]
            #print("page_list")
            topic_name = df_predict_question.iloc[i]['predicted_topic']  
            #print(topic_name)
            add_question_to_pdf(file_name, page_list, output_folder + '/' + topic_name)
        
        print("\n\n")
        print(df_predict_question)
        
        return df_predict_question
    except:
        print('Did not manage to predict for ' + file_name)

def remove_temp(output_folder):
    # Get a list of all the file paths that ends with .txt from in specified directory
    fileList = glob.glob(output_folder + '/*temporary.pdf')

    # Iterate over the list of filepaths & remove each file.
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)
    print("Temporary files removed.")

def zip_folder(output_folder):
    print("Zipping sorted files.")
    with ZipFile('tmp/sorted_output.zip', 'w') as zipObj:
       # Iterate over all the files in directory
       for folderName, subfolders, filenames in os.walk(output_folder):
           for filename in filenames:
               #create complete filepath of file in directory
               filePath = os.path.join(folderName, filename)
               # Add file to zip
               zipObj.write(filePath)
    print("Files are zipped.")
               
# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

# This route will show a form to perform an AJAX request
# jQuery is loaded to execute the request and update the
# value of the operation
@app.route('/')
def index():
    return render_template('index.html')


# Route that will process the file upload
@app.route('/upload', methods=['POST'])
def upload():
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    # Get the name of the uploaded files
    uploaded_files = request.files.getlist("file[]")
    filenames = []
    for file in uploaded_files:
        # Check if the file is one of the allowed types/extensions
        if file and allowed_file(file.filename):
            # Make the filename safe, remove unsupported chars
            filename = secure_filename(file.filename)
            # Move the file form the temporal folder to the upload
            # folder we setup
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # Save the filename into a list, we'll use it later
            filenames.append(filename)
            # Redirect the user to the uploaded_file route, which
            # will basicaly show on the browser the uploaded file
    df_all_predicted = pd.DataFrame()
    
    filenames.reverse()
    
    '''# for testing and debugging
    #html_string = pdf_to_html('D:/CIE_Machine_Learning/data/9702_backup_new_renamed/2007_11Nov_P2_9702_ms.pdf')
    html_string = pdf_to_html('tmp/' + filenames[0])
    pages_text = get_pages(html_string)
    question_numbers_corrected = find_question_in_html(html_string)
    question_text_notags = get_question_texts(question_numbers_corrected, html_string)
    #question_numbers_corrected[10]
    #get_question_marks(question_text_notags[0])
    get_pages_of_questions(pages_text, question_numbers_corrected, html_string)'''
    
    for file in filenames:
        file = 'tmp/' + file
        df = predict_by_question(file, output_folder)
        df_all_predicted = pd.concat([df_all_predicted, df])
        
        #template.render(filenames=[file])
    remove_temp(output_folder)
    zip_folder(output_folder)
    df_all_predicted.to_excel('tmp/report.xlsx')
    
    # Load an html page with a link to each uploaded file
    return render_template('sorted.html', filenames=filenames, sorted_output = "sorted_output.zip", report_output = "report.xlsx")	
    	

# This route is expecting a parameter containing the name
# of a file. Then it will locate that file on the upload
# directory and show it on the browser, so if the user uploads
# an image, that image is going to be show after the upload
@app.route('/tmp/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route('/tmp/<filename>')
def sorted_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=int("80"),
        debug=True
    )
