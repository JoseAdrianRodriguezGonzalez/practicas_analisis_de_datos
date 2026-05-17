from lxml import etree
import requests
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
# curl -L -X GET -H "Accept: application/xml" "https://boe.es/datosabiertos/api/boe/sumario/20260202"
def API_call(path:str,mode:str,headers:dict[str,str]):
    if mode=="get":
        r=requests.get(path,headers=headers)
        try:
            print(f"Codigo de respuesta:{r}")
            with open("data.xml","wb") as f:
                f.write(r.content)
        except:
            raise ValueError(f"Codigo de error {r.status_code}")
def load_documents(path:str)->etree:
    tree=etree.parse(path)    
    return tree 
def verify_existence(lst):
    return lst[0] if lst else None
def explore_file(tree:etree)->pd.DataFrame:
    fecha=tree.xpath("//fecha_publicacion/text()")[0]
    publicacion=tree.xpath("//publicacion/text()")[0]
    numero=tree.xpath("//diario/@numero")[0]
    items=tree.xpath("//item")
    dataset=[]
    for item in items:
        row={
            "id":verify_existence(item.xpath("identificador/text()")),
            "titulo":verify_existence(item.xpath("titulo/text()")),
            "seccion":verify_existence(item.xpath("ancestor::seccion/@nombre")),
            "departamento":verify_existence(item.xpath("ancestor::departamento/@nombre")),
            "epigrafe":verify_existence(item.xpath("ancestor::epigrafe/@nombre")),
            "url_pdf":verify_existence(item.xpath("url_pdf/text()")),
            "url_html":verify_existence(item.xpath("url_html/text()")),
            "fecha":fecha,
            "publicacion":publicacion,
            "numero":numero
        }
        dataset.append(row)
    df=pd.DataFrame(dataset)
    return df
def analisis(df:pd.DataFrame):
    patron = r"^(Ley|Real Decreto|Decreto|Orden|Resolución|Acuerdo|Anuncio|Enmienda|Convenio|Tratado)"
    print(df["seccion"].value_counts())
    df["tipo"]=df["titulo"].str.extract(patron,expand=False).fillna("Otro")
    df.loc[df["seccion"].str.contains("IV. Administración de Justicia", na=False),
       "tipo"] = "Judicial"
    freq=df["tipo"].value_counts()
    rel=df["tipo"].value_counts(normalize=True)*100
    print(freq)
    table=pd.DataFrame({
        "frecuencia":freq,
        "porcentaje":rel.round(2)
    })
    return table
def grafica_barras(freq,path):
    plt.figure()
    freq.plot(kind="bar")
    plt.title("Frecuencia por tipo de legislación")
    plt.xlabel("Tipo")
    plt.ylabel("Número de documentos")
    plt.savefig(path)
    plt.show()
def graficar_pastel(freq,path):
    fig, ax = plt.subplots(figsize=(7, 8))
    wedges, _ = ax.pie(freq,startangle=90)
    ax.set_title("Distribución por tipo de legislación",pad=30)
    plt.subplots_adjust(top=0.80) 
    # Colocar etiquetas afuera con líneas
    for i, w in enumerate(wedges):
        ang = (w.theta2 - w.theta1)/2 + w.theta1
        x = np.cos(np.deg2rad(ang))
        y = np.sin(np.deg2rad(ang))
        offset=0.1*(i-len(freq)/2)
        ax.annotate(
            f"{freq.index[i]} ({freq.iloc[i]}) %",
            xy=(x, y),
            xytext=(1.4 * x+(offset+0.1), 1 * y+(offset+0.1)),  # posición afuera
            arrowprops=dict(arrowstyle="-"),
            ha="left"
        )
    plt.savefig(path)
    plt.show()
def main():
    header={
        "Accept":"application/xml"
    }
#    API_call("https://boe.es/datosabiertos/api/boe/sumario/20260202","get",headers=header)
    #first step, load the documents from the origin
    tree=load_documents("data.xml")
    df=explore_file(tree)
    df.to_csv("leyes.csv")
    table=analisis(df)

    grafica_barras(table["frecuencia"],"grafica_barras.png")
    graficar_pastel(table["porcentaje"],"grafica_pastel.png")
main()

