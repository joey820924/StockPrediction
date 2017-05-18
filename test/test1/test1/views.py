from django.http import HttpResponse
from svmutil import *
import os
os.chdir('/Users/joey/libsvm-3.22/tools/')

def here(request):
    return HttpResponse('Kobe Bryant')

def add(request, a, b):
    s = int(a)+int(b)
    return HttpResponse(str(s))

def convert(label,a1,a2,a3,a4,a5,a6,a7,a8):
    o = open( "/Users/joey/Documents/stockcal/tmp/tmp.txt", 'wb' )
    line = []
    line.append(a1)
    line.append(a2)
    line.append(a3)
    line.append(a4)
    line.append(a5)
    line.append(a6)
    line.append(a7)
    line.append(a8)
    new_line = construct_line( label, line )
    o.write( new_line )

def construct_line( label, line ):
    new_line = []
    if float( label ) == 0.0:
        label = "0"
    new_line.append( str(float(label) ))

    for i, item in enumerate( line ):
        new_item = "%s:%s" % ( i + 1, item )
        new_line.append( new_item )
    new_line = " ".join( new_line )
    new_line += "\n"
    return new_line
    
    

def getPrice(request,username = None):
    code = request.GET.get("code")
    label = request.GET.get("label")
    macd = request.GET.get("macd")
    macds = request.GET.get("macds")
    macdh = request.GET.get("macdh")
    kdjk = request.GET.get("kdjk")
    kdjd = request.GET.get("kdjd")
    kdjj = request.GET.get("kdjj")
    rsi5 = request.GET.get("rsi5")
    rsi10 = request.GET.get("rsi10")
    MP = "/Users/joey/Documents/stockcal/model/"+code+".txt"
    m = svm_load_model(MP)
    convert(label,macd,macds,macdh,kdjk,kdjd,kdjj,rsi5,rsi10)
    params = "svm-scale -s /Users/joey/Documents/stockcal/tmp/rule.txt /Users/joey/Documents/stockcal/tmp/tmp.txt >/Users/joey/Documents/stockcal/tmp/scale.txt"
    os.system(params)
    y,x = svm_read_problem("/Users/joey/Documents/stockcal/tmp/scale.txt")
    p_label, p_acc, p_val = svm_predict(y, x, m)
    return HttpResponse(p_label)



# ---


