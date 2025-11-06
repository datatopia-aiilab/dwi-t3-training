## เป้าหมาย: พัฒนาโมเดล Deep Learning ที่มีความแม่นยำสูง (Dice \> 95%) สำหรับการ Segment รอยโรค Ischemic Stroke จากภาพ 2D DWI (B-1000)

## ปัญหาที่ต้องแก้: โมเดล U-Net ปัจจุบัน (Dice 75%) ล้มเหลวในการตรวจจับส่วนที่สว่างจาง (Faint) ของรอยโรค และจับได้เฉพาะส่วนที่สว่างเข้ม (Bright)

## กลยุทธ์ที่เลือก (The Definitive Path): เราจะใช้กลยุทธ์ผสมผสาน 4 แกนหลักที่ทำงานร่วมกันอย่างเป็นระบบ (A 4-Core Integrated Strategy) ได้แก่:

1. ## 2.5D Input: ใช้บริบท 3 มิติ (Slices ข้างเคียง) มาช่วยในการตัดสินใจของโมเดล 2 มิติ

2. ## Contrast Enhancement (CLAHE): "เร่ง" สัญญาณของส่วนที่จางให้ชัดเจนขึ้นในขั้นตอน Preprocessing

3. ## Attention U-Net: "บังคับ" ให้โมเดล "เพ่งความสนใจ" ไปยังพื้นที่ที่ซับซ้อนและทายยาก (เช่น ขอบเขตที่จางๆ)

4. ## Combo Loss (Focal \+ Dice): "ลงโทษ" โมเดลอย่างหนักเมื่อทายพิกเซลที่ทายยาก (Faint pixels) พลาด และจัดการปัญหา Imbalance

## กลยุทธ์นี้ถูกออกแบบมาเพื่อเปลี่ยนจุดอ่อนของโมเดล (การแยกแยะความสว่างต่ำ) ให้กลายเป็นจุดแข็ง

##  **พิมพ์เขียวรายขั้นตอน (Step-by-Step Blueprint)**

### **ขั้นตอนที่ 0: การแบ่งข้อมูล (Data Splitting)**

* **การกระทำ:** ก่อนเริ่มทุกอย่าง ให้แบ่งรายชื่อไฟล์ 900 ภาพ ออกเป็น 3 ส่วน: Training (70%), Validation (15%), และ Test (15%)  
* **เหตุผล:** เราต้องมั่นใจว่า Test Set เป็นข้อมูลที่โมเดล "ไม่เคยเห็น" เลยในทุกกระบวนการ (รวมถึง Preprocessing) เพื่อการวัดผลที่ยุติธรรม

### **ขั้นตอนที่ 1: การประมวลผลข้อมูลล่วงหน้า (01\_preprocess.py)**

นี่คือขั้นตอนการแปลงข้อมูลจาก 1\_data\_raw ไปยัง 2\_data\_processed

1. **Iterate & Load:** วนลูปไฟล์ทั้งหมดใน Train, Val, Test set  
2. **Resize (ถ้าจำเป็น):** หากภาพมีขนาดไม่เท่ากัน ให้ Resize ทุกภาพ (และ Mask) ให้มีขนาดมาตรฐานเดียวกัน (เช่น 256x256) โดยใช้ bilinear (สำหรับภาพ) และ nearest (สำหรับ Mask)  
3. **CLAHE (Contrast Limited Adaptive Histogram Equalization):**  
   * **การกระทำ:** ใช้ skimage.exposure.equalize\_adapthist กับภาพ DWI  
   * **เหตุผล (Why):** นี่คือหัวใจของการแก้ปัญหา\! CLAHE จะเพิ่ม Contrast ในระดับ Local ทำให้พิกเซลที่ "จาง" (Faint) ถูก "เร่ง" ให้ชัดเจนขึ้นมา โดยไม่ทำให้ส่วนที่ "สว่าง" (Bright) อยู่แล้วจ้าจนเสียรายละเอียด  
4. **Normalization (Z-score):**  
   * **การกระทำ:** คำนวณค่า mean และ std *จาก Training set เท่านั้น* จากนั้นนำค่านี้ไปใช้ Normalization (Z-score) กับทุก set (Train, Val, Test)  
   * **เหตุผล (Why):** ทำให้ข้อมูลมี mean=0, std=1 ซึ่งจำเป็นต่อการเรียนรู้ที่เสถียรของโมเดล  
5. **Save as .npy:** บันทึกภาพที่ผ่านการประมวลผลแล้ว (และ Mask) ลงในโฟลเดอร์ 2\_data\_processed โดยแยก Train/Val/Test  
   * **เหตุผล (Why):** การทำ CLAHE และ Normalization ใช้เวลา การคำนวณล่วงหน้าและบันทึกเป็น .npy (NumPy array) จะทำให้การโหลดข้อมูลตอนเทรน *เร็วมาก*

### **ขั้นตอนที่ 2: การสร้างคลาสชุดข้อมูล 2.5D (dataset.py)**

เราจะสร้าง PyTorch Dataset ที่จะป้อนข้อมูลแบบ **2.5D** ให้โมเดล

1. **\_\_init\_\_(self, image\_paths, mask\_paths, augmentations=None):**  
   * รับรายการไฟล์ภาพและ Mask (จาก 2\_data\_processed)  
   * เก็บ augmentations (เช่น Albumentations) ไว้  
2. **\_\_len\_\_(self):** คืนค่าจำนวนภาพทั้งหมด  
3. **\_\_getitem\_\_(self, idx):** นี่คือส่วนที่ซับซ้อนที่สุด:  
   * **ก. โหลดภาพเป้าหมาย (Slice N):** img\_n \= np.load(self.image\_paths\[idx\])  
   * **ข. โหลด Mask เป้าหมาย:** mask\_n \= np.load(self.mask\_paths\[idx\]) (และแปลงเป็น float32)  
   * **ค. โหลดภาพข้างเคียง (Slice N-1 และ N+1):**  
     * เราต้องหาวิธี "อนุมาน" ชื่อไฟล์ของ Slice ข้างเคียง (เช่น จากชื่อไฟล์)  
     * **Slice N-1:** img\_n\_minus\_1 \= np.load(path\_to\_n\_minus\_1)  
     * **Slice N+1:** img\_n\_plus\_1 \= np.load(path\_to\_n\_plus\_1)  
     * **การจัดการขอบ (Edge Handling):** ถ้า idx คือ Slice แรก (ไม่มี N-1) หรือ Slice สุดท้าย (ไม่มี N+1) ให้สร้าง Array 0 (ภาพสีดำ) ที่มีขนาดเท่ากันมาแทนที่  
   * **ง. สร้าง 2.5D Input:**  
     * stacked\_image \= np.stack(\[img\_n\_minus\_1, img\_n, img\_n\_plus\_1\], axis=-1)  
     * *หมายเหตุ: เราใช้ axis=-1 เพื่อให้ได้ Shape (Height, Width, 3\)*  
   * **จ. Data Augmentation (On-the-fly):**  
     * if self.augmentations:  
     * augmented \= self.augmentations(image=stacked\_image, mask=mask\_n)  
     * **ท่าที่แนะนำ:** HorizontalFlip, Rotate (เล็กน้อย), ElasticTransform (สำคัญมาก), RandomBrightnessContrast (เล็กน้อย)  
   * **ฉ. แปลงเป็น Tensor:** return image\_tensor, mask\_tensor (อย่าลืม Permute image\_tensor ให้ Channel อยู่มิติแรก (3, H, W))  
* **เหตุผล (Why 2.5D):** การให้โมเดลเห็น Slices ข้างเคียง (N-1, N+1) จะให้ "บริบท 3 มิติ" ที่สำคัญมาก ถ้าโมเดลเห็นจุดจางๆ บน Slice N มันจะมั่นใจมากขึ้นว่านี่คือ "รอยโรคจริง" หากมันเห็นความต่อเนื่องบน N-1 และ N+1 ด้วย

### **ขั้นตอนที่ 3: สถาปัตยกรรมโมเดล (model.py)**

เราจะสร้าง **Attention U-Net**

1. **Input Layer:** Convolution layer แรก ต้องรับ in\_channels=3 (ไม่ใช่ 1\) เพื่อรองรับ 2.5D Input ของเรา  
2. **Encoder (Down-sampling):** สร้างตาม U-Net มาตรฐาน (Conv blocks \+ Max Pooling)  
3. **Attention Gate (AG) Module:**  
   * สร้าง Class AttentionGate แยกต่างหาก  
   * **Input:** g (จาก Decoder) และ x (จาก Skip Connection)  
   * **Process:**  
     1. g \-\> Conv 1x1  
     2. x \-\> Conv 1x1  
     3. บวก g และ x ที่ผ่าน Conv แล้ว  
     4. ผ่าน ReLU \-\> Conv 1x1 \-\> Sigmoid (ได้ attention\_map ค่า 0-1)  
     5. return x \* attention\_map (คูณ attention\_map กลับเข้าไปที่ x เดิม)  
   * **เหตุผล (Why AG):** AG จะเรียนรู้ที่จะ "กรอง" ข้อมูลที่วิ่งผ่าน Skip Connection มันจะ "ลดความสำคัญ" ของ Background และ "เน้น" เฉพาะส่วนที่คล้ายรอยโรค (รวมถึงส่วนที่จาง) ก่อนส่งไปให้ Decoder  
4. **Decoder (Up-sampling):**  
   * ในแต่ละขั้นของ Decoder, *ก่อน* ที่จะ torch.cat (เชื่อม) Skip Connection:  
   * x\_att \= self.attention\_gate(g=g\_from\_decoder, x=x\_from\_encoder)  
   * merged \= torch.cat(\[g\_from\_decoder\_upsampled, x\_att\], dim=1)  
   * ... (ต่อด้วย Conv Blocks)  
5. **Output Layer:** Conv 1x1 ตามด้วย Sigmoid (เพื่อให้ได้ Mask 0-1)

### **ขั้นตอนที่ 4: นิยามฟังก์ชันสูญเสีย (loss.py)**

เราจะสร้าง **Combo Loss** เพื่อจัดการปัญหา Imbalance และ Hard Examples

1. **Focal Loss:**  
   * **การกระทำ:** สร้าง Class FocalLoss(alpha=0.25, gamma=2.0)  
   * **เหตุผล (Why):** gamma=2.0 จะ "ลดทอน" Loss จากพิกเซลที่ทายถูกอยู่แล้ว (Easy Examples) และ "บังคับ" ให้โมเดลไปโฟกัสที่พิกเซลที่ทายยาก (Hard Examples) ซึ่งก็คือ "ส่วนที่จาง" ที่เรากำลังมีปัญหานั่นเอง\!  
2. **Dice Loss:**  
   * **การกระทำ:** สร้าง Class DiceLoss()  
   * **เหตุผล (Why):** วัดค่า Overlap โดยตรง เหมาะกับ Imbalance Segmentation  
3. **Combo Loss:**  
   * **การกระทำ:** สร้าง Class ComboLoss(focal\_weight=0.5, dice\_weight=0.5)  
   * def forward(self, pred, target):  
   * loss \= (self.focal\_weight \* self.focal\_loss(pred, target)) \+ (self.dice\_weight \* self.dice\_loss(pred, target))  
   * return loss

### **ขั้นตอนที่ 5: การฝึกโมเดล (train.py)**

นี่คือ Script "ผู้ควบคุม" (Orchestrator)

1. **Setup:** โหลด config.py, สร้าง DataLoader (สำหรับ Train และ Val), โหลด model (Attention U-Net), โหลด loss\_fn (Combo Loss)  
2. **Optimizer:** torch.optim.AdamW(model.parameters(), lr=...)  
3. **Scheduler:** torch.optim.lr\_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.1)  
   * **เหตุผล (Why):** เราจะ "ลด Learning Rate" อัตโนมัติ หาก "Validation Dice Score" (mode='max') ไม่ดีขึ้นเป็นเวลา 5 epochs  
4. **Training Loop:**  
   * for epoch in range(num\_epochs):  
   * train\_one\_epoch(...) \-\> คืนค่า avg\_train\_loss, avg\_train\_dice  
   * validate\_one\_epoch(...) \-\> คืนค่า avg\_val\_loss, avg\_val\_dice  
   * scheduler.step(avg\_val\_dice) (ใช้ Val Dice ในการตัดสินใจ)  
   * **Logging:** พิมพ์ผลลัพธ์ของ Epoch (Train/Val Loss, Train/Val Dice)  
   * **Checkpointing:**  
     * if avg\_val\_dice \> best\_dice\_so\_far:  
     * best\_dice\_so\_far \= avg\_val\_dice  
     * torch.save(model.state\_dict(), '3\_model\_weights/best\_attention\_unet\_model.pth')  
     * **เหตุผล:** เราจะบันทึกเฉพาะโมเดลที่ทำ "Validation Dice" ได้ดีที่สุดเท่านั้น

### **ขั้นตอนที่ 6: การประเมินผลและการแสดงภาพ (evaluate.py)**

นี่คือขั้นตอน "พิสูจน์ผลลัพธ์"

1. **Load Best Model:** โหลด best\_attention\_unet\_model.pth  
2. **Load Test Set:** สร้าง DataLoader สำหรับ test (ไม่ต้องใช้ Augmentation)  
3. **Run Inference:** model.eval(), with torch.no\_grad():, วนลูป Test Set และคำนวณ Dice Score, Precision, Recall ทั้งหมด  
4. **Plot Training Curves:**  
   * **การกระทำ:** โหลดประวัติการเทรน (ที่ควรบันทึกไว้) และพล็อต 2 กราฟ:  
     1. Loss vs. Epochs (Train Loss vs. Val Loss)  
     2. Dice vs. Epochs (Train Dice vs. Val Dice)  
   * **เหตุผล (Why):** เพื่อตรวจจับ Overfitting และแสดงความเสถียรของการเทรน (ใช้สีที่มืออาชีพ เช่น น้ำเงิน/ส้ม หรือ น้ำเงิน/แดง)  
5. **Generate Qualitative Results (สำคัญที่สุด):**  
   * **การกระทำ:** สุ่มตัวอย่าง 10 ภาพจาก Test Set  
   * สำหรับแต่ละตัวอย่าง, สร้างภาพ 1x3 (Original, Ground Truth, Prediction)  
   * **คอลัมน์ 1 (Original):** แสดงภาพ original\_dwi (ควรเป็น Slice N ที่เป็น 2D)  
   * **คอลัมน์ 2 (Ground Truth):** แสดง original\_dwi และซ้อนทับ ground\_truth\_mask ด้วยสีแดง (Alpha=0.5)  
   * **คอลัมน์ 3 (Prediction):** แสดง original\_dwi และซ้อนทับ predicted\_mask (ที่ได้จากโมเดล) ด้วยสีฟ้า/เขียว (Alpha=0.5)  
   * **เหตุผล (Why):** นี่คือการพิสูจน์เชิงคุณภาพว่าโมเดล "แก้ปัญหา" ได้หรือไม่ เราคาดหวังจะเห็น Mask สีฟ้า (Prediction) ครอบคลุมพื้นที่ "จาง" (Faint) ได้ใกล้เคียงกับ Mask สีแดง (Ground Truth)

---

## **4\. บทสรุปความคาดหวัง**

Blueprint นี้ไม่ได้แก้ปัญหาด้วยเทคนิคเดียว แต่เป็นการสร้าง "ระบบ" ที่ทุกส่วนทำงานสอดประสานกัน: **CLAHE** เผยจุดที่ซ่อนเร้น, **2.5D** ให้บริบท, **Attention U-Net** บังคับให้โฟกัส และ **Combo Loss** ลงโทษเมื่อทำงานพลาดในจุดที่ยาก

ผลลัพธ์ที่คาดหวังคือโมเดลที่ "มองเห็น" รอยโรคได้ครบถ้วนสมบูรณ์ (ทั้งส่วนที่จางและเข้ม) ส่งผลให้ Dice Score เพิ่มขึ้นอย่างมีนัยสำคัญ

