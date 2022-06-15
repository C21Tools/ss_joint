import cv2
import numpy as np
import streamlit as st
from operator import attrgetter
from PIL import Image


def main():
    st.markdown('# C21アセン画像結合ツール')
    st.markdown('## 使い方')
    st.markdown('- 結合したいスクリーンショットをまとめてドラッグアンドドロップしてください。パーツツリーとステータスが結合された画像が生成されます。')
    st.markdown('- スクリーンショットの撮影方法はNote(https://note.com/take_c21/n/nf7fc20d5558e)を参照してください。')
    st.markdown('- スクリーンショットは画像処理のため、Streamlit cloudのサーバーにいったん送られますが、サーバーに保存されることはなく、ブラウザを閉じると消去されます。')
    uploaded_files = st.file_uploader("↓にスクリーンショットをドラッグアンドドロップ", accept_multiple_files=True)
    if uploaded_files:    
        files_img=sorted(uploaded_files, key=attrgetter('name'))
        files_template=np.roll(files_img,-1)
        frame = Image.open(files_img[-1])
        frame = np.array(frame)
        header = frame[71:90,18:211]
        footer = frame[90:408,18:211]
        result = header

        for file_img,file_template in zip(files_img[:-1],files_template[:-1]):
            img = Image.open(file_img)
            img = np.array(img)
            img = img[90:297,18:211]
            template = Image.open(file_template)
            template = np.array(template)
            template = template[90:100,18:211]
            match = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(match)
            img = img[0:maxLoc[1],:]
            result = cv2.vconcat([result,img])

        result = cv2.vconcat([result,footer])

        status = frame[71:408,234:790]
        status = cv2.copyMakeBorder(status, 0, result.shape[0]-status.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0)) 

        result = cv2.hconcat([result,status])
        st.image(result)
            
if __name__ == '__main__':
    main()
