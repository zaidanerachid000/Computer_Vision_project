import 'dart:async';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  final firstCamera = cameras.firstWhere(
    (camera) => camera.lensDirection == CameraLensDirection.back,
    orElse: () => cameras.first,
  );

  runApp(MyApp(camera: firstCamera));
}

class MyApp extends StatelessWidget {
  final CameraDescription camera;

  const MyApp({super.key, required this.camera});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Object Detection',
      theme: ThemeData.dark(),
      home: CameraScreen(camera: camera),
    );
  }
}

class CameraScreen extends StatefulWidget {
  final CameraDescription camera;

  const CameraScreen({super.key, required this.camera});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  late CameraController _controller;
  late Future<void> _initializeControllerFuture;
  late WebSocketChannel _channel;
  
  bool _isDetecting = false;
  Timer? _timer;
  List<dynamic> _detections = [];

  // Changer localhost par l'IP de la machine locale exécutant Docker (ex: 192.168.1.x)
  final String websocketUrl = 'ws://192.168.1.11:8000/ws/detect';

  @override
  void initState() {
    super.initState();
    _controller = CameraController(
      widget.camera,
      ResolutionPreset.medium,
      enableAudio: false,
    );
    _initializeControllerFuture = _controller.initialize();
    
    _connectWebSocket();
  }

  void _connectWebSocket() {
    try {
      _channel = WebSocketChannel.connect(Uri.parse(websocketUrl));
      _channel.stream.listen(
        (message) {
          final data = jsonDecode(message);
          if (data['detections'] != null) {
            setState(() {
              _detections = data['detections'];
            });
          }
        },
        onError: (error) {
          print('WebSocket Error: $error');
        },
        onDone: () {
          print('WebSocket Closed');
          // Tentative de reconnexion après 5 secondes
          Future.delayed(const Duration(seconds: 5), _connectWebSocket);
        },
      );
    } catch (e) {
      print('WebSocket Connection Error: $e');
    }
  }

  void _startDetection() {
    if (_isDetecting) return;
    setState(() => _isDetecting = true);
    
    // Envoyer une image toutes les 500 ms pour analyse
    _timer = Timer.periodic(const Duration(milliseconds: 500), (timer) async {
      try {
        await _initializeControllerFuture;
        final image = await _controller.takePicture();
        final bytes = await image.readAsBytes();
        final base64Image = base64Encode(bytes);
        
        // Envoyer au WebSocket
        _channel.sink.add(base64Image);
      } catch (e) {
        print("Erreur de capture : $e");
      }
    });
  }

  void _stopDetection() {
    setState(() {
      _isDetecting = false;
      _detections = [];
    });
    _timer?.cancel();
  }

  @override
  void dispose() {
    _controller.dispose();
    _timer?.cancel();
    _channel.sink.close();
    super.dispose();
  }

  List<Widget> _buildBoundingBoxes(BoxConstraints constraints) {
    if (_detections.isEmpty) return [];

    // Obtenir la taille de l'aperçu caméra
    final previewSize = _controller.value.previewSize!;
    
    // Ratio pour convertir les coordonnées de l'image d'origine à la taille de l'écran
    final scaleX = constraints.maxWidth / previewSize.height;
    final scaleY = constraints.maxHeight / previewSize.width;

    return _detections.map((detection) {
      final color = detection['color'] == 'green' ? Colors.green : Colors.red;
      
      return Positioned(
        left: detection['x'] * scaleX,
        top: detection['y'] * scaleY,
        width: detection['width'] * scaleX,
        height: detection['height'] * scaleY,
        child: Container(
          decoration: BoxDecoration(
            border: Border.all(color: color, width: 3),
          ),
          child: Align(
            alignment: Alignment.topLeft,
            child: Container(
              color: color,
              padding: const EdgeInsets.all(4),
              child: Text(
                detection['label'],
                style: const TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                  fontSize: 12,
                ),
              ),
            ),
          ),
        ),
      );
    }).toList();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Détection d\'objets'),
        actions: [
          IconButton(
            icon: Icon(_isDetecting ? Icons.stop : Icons.play_arrow),
            color: _isDetecting ? Colors.red : Colors.green,
            onPressed: _isDetecting ? _stopDetection : _startDetection,
          )
        ],
      ),
      body: FutureBuilder<void>(
        future: _initializeControllerFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.done) {
            return LayoutBuilder(
              builder: (BuildContext context, BoxConstraints constraints) {
                return Stack(
                  fit: StackFit.expand,
                  children: [
                    CameraPreview(_controller),
                    ..._buildBoundingBoxes(constraints),
                  ],
                );
              },
            );
          } else {
            return const Center(child: CircularProgressIndicator());
          }
        },
      ),
    );
  }
}
