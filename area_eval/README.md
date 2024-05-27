# Area eval lib

New metrics to evaluate area predictions model

## List of new metrics

* > __Area Relative Ratio__ (ARR)
    $$ARR_i = \frac{A_{real, i}-A_{pred, i}}{A_{real, i}}$$

* > __mean Area Relative Ratio__ (mARR)
    $$mARR = \frac{1}{n}\sum_{i=0}^n \frac{abs(A_{real, i}-A_{pred, i})}{A_{real, i}}$$

* > __Prediction Coverage Index__ (PCI)
    $$ PCI = \frac{len(\{i\text{ t.c. } \forall i \in [0, ..., n ] len(A_{real,i})==len(A_{pred,i}) \})}{len(A_{real})} $$

* > __False Area Prediction Relative Rate__ (FAPRR)
    $$FAPRR_{\alpha} = len \Bigg(\frac{abs(A_{real, i}-A_{pred, i})}{A_{real, i}} > \alpha\Bigg) \hspace{1cm}\text{with } i \in [0,...,n]$$

* > __Area Difference Over Mean Real Area percentage error__ (AOM)
    $$AOM =  \frac{\frac{1}{n}\sum_{i=0}^n abs(A_{real, i}-A_{pred, i})}{\frac{1}{n}\sum_{i=0}^n A_{real, i}} 100$$