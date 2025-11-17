PVObject_=pvModule
  Version=7.3.1
  Flags=$0043

  PVObject_Commercial=pvCommercial
    Comment=Demo 450W Half-Cut
    Flags=$0041
    Manufacturer=Demo Solar Inc.
    Model=Demo 450W Half-Cut
    DataSource=Calculated
    YearBeg=2025
    Width=1.926
    Height=3.192
    Depth=0.035
    Weight=70.7
    NPieces=240
    PriceDate=17/11/25
  End of PVObject pvCommercial

  Technol=mtSiMono
  NCelS=120
  NCelP=2
  NDiode=3

  GRef=1000
  TRef=25.0
  PNom=648.0
  PNomTolLow=3.0
  PNomTolUp=3.0
  Isc=10.300
  Voc=81.60
  Imp=9.552
  Vmp=67.84

  muISC=0.000500
  muVocSpec=-0.280
  muPmpReq=-0.350

  TModule_NOCT=45.0
  T_NOCT=45.0
  G_NOCT=800
  muPmpReq_NOCT=-0.350

  RShunt=30000.0
  Rp_0=45000.0
  Rp_Exp=5.50
  RSerie=0.5400
  Gamma=1.005
  muGamma=-0.000300

  PVObject_IAM=pvIAM
    Flags=$00
    IAMMode=UserProfile
    IAMProfile=TCubicProfile
       0  1.0000
      10  1.0000
      20  0.9995
      30  0.9985
      40  0.9960
      50  0.9900
      60  0.9750
      70  0.9350
      75  0.8950
      80  0.8200
      85  0.6800
      90  0.0000
  End of PVObject pvIAM

  OperPoints=0

End of PVObject pvModule