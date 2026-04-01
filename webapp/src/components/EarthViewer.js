import React, { useEffect, useRef } from "react";
import * as Cesium from "cesium";
import * as satellite from "satellite.js";
import "cesium/Build/Cesium/Widgets/widgets.css";

const CAMERA_DESTINATION = Cesium.Cartesian3.fromDegrees(77, 13, 20000000);
const ORBIT_MINUTES = 90;
const POSITION_UPDATE_INTERVAL_MS = 250;

const EarthViewer = ({ objects, onObjectClick }) => {
  const containerRef = useRef(null);
  const viewerRef = useRef(null);
  const handlerRef = useRef(null);
  const onObjectClickRef = useRef(onObjectClick);
  const orbitEntityRef = useRef(null);
  const satelliteEntriesRef = useRef([]);
  const lastPositionUpdateRef = useRef(0);

  useEffect(() => {
    onObjectClickRef.current = onObjectClick;
  }, [onObjectClick]);

  useEffect(() => {
    if (!containerRef.current) {
      return undefined;
    }

    const token = process.env.REACT_APP_CESIUM_TOKEN;
    const hasIonToken = Boolean(token && token !== "your_token_here");

    if (hasIonToken) {
      Cesium.Ion.defaultAccessToken = token;
    }

    const viewer = new Cesium.Viewer(containerRef.current, {
      infoBox: false,
      selectionIndicator: false,
      animation: false,
      timeline: false,
      geocoder: false,
      homeButton: false,
      navigationHelpButton: false,
      fullscreenButton: false,
      sceneModePicker: false,
      baseLayerPicker: hasIonToken,
      baseLayer: hasIonToken
        ? undefined
        : Cesium.ImageryLayer.fromProviderAsync(
            Cesium.TileMapServiceImageryProvider.fromUrl(
              Cesium.buildModuleUrl("Assets/Textures/NaturalEarthII")
            )
          ),
    });

    viewer.scene.globe.depthTestAgainstTerrain = true;
    viewer.scene.globe.enableLighting = true;
    viewer.scene.logarithmicDepthBuffer = true;

    viewer.camera.setView({
      destination: CAMERA_DESTINATION,
    });

    viewer.clock.shouldAnimate = true;
    viewerRef.current = viewer;

    const updateSelectedObject = (entry) => {
      if (!entry || !onObjectClickRef.current) {
        return;
      }

      onObjectClickRef.current({
        id: entry.entity.name ?? entry.entity.id,
        type: entry.objectType,
        altitude: Number(entry.entity.currentAltitudeKm ?? 0).toFixed(2),
        velocity: Number(entry.entity.currentVelocityKmS ?? 0).toFixed(3),
      });
    };

    const drawOrbit = (entry) => {
      if (orbitEntityRef.current) {
        viewer.entities.remove(orbitEntityRef.current);
        orbitEntityRef.current = null;
      }

      if (!entry?.satrec) {
        return;
      }

      const positions = [];
      const now = new Date();

      for (let minute = 0; minute < ORBIT_MINUTES; minute += 1) {
        const time = new Date(now.getTime() + minute * 60000);
        const pv = satellite.propagate(entry.satrec, time);

        if (!pv?.position) {
          continue;
        }

        const gmst = satellite.gstime(time);
        const geo = satellite.eciToGeodetic(pv.position, gmst);

        if (!geo) {
          continue;
        }

        const lon = Cesium.Math.toDegrees(geo.longitude);
        const lat = Cesium.Math.toDegrees(geo.latitude);
        const alt = geo.height * 1000;

        if (
          Number.isNaN(lon) ||
          Number.isNaN(lat) ||
          Number.isNaN(alt)
        ) {
          continue;
        }

        positions.push(Cesium.Cartesian3.fromDegrees(lon, lat, alt));
      }

      if (positions.length < 2) {
        return;
      }

      orbitEntityRef.current = viewer.entities.add({
        polyline: {
          positions,
          width: 2,
          material: Cesium.Color.YELLOW,
          arcType: Cesium.ArcType.NONE,
        },
      });
    };

    const updateSatellitePositions = () => {
      const nowMs = Date.now();

      if (nowMs - lastPositionUpdateRef.current < POSITION_UPDATE_INTERVAL_MS) {
        return;
      }

      lastPositionUpdateRef.current = nowMs;
      const now = new Date(nowMs);

      satelliteEntriesRef.current.forEach((entry) => {
        const pv = satellite.propagate(entry.satrec, now);

        if (!pv?.position) {
          return;
        }

        const gmst = satellite.gstime(now);
        const geo = satellite.eciToGeodetic(pv.position, gmst);

        if (!geo) {
          return;
        }

        const lon = Cesium.Math.toDegrees(geo.longitude);
        const lat = Cesium.Math.toDegrees(geo.latitude);
        const alt = geo.height * 1000;

        if (
          Number.isNaN(lon) ||
          Number.isNaN(lat) ||
          Number.isNaN(alt)
        ) {
          return;
        }

        entry.entity.position = Cesium.Cartesian3.fromDegrees(lon, lat, alt);
        entry.entity.currentAltitudeKm = geo.height;

        if (pv.velocity) {
          const { x, y, z } = pv.velocity;
          entry.entity.currentVelocityKmS = Math.sqrt(
            x * x + y * y + z * z
          );
        }

        if (viewer.selectedEntity === entry.entity) {
          updateSelectedObject(entry);
        }
      });
    };

    viewer.clock.onTick.addEventListener(updateSatellitePositions);

    const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
    handlerRef.current = handler;

    handler.setInputAction((click) => {
      const picked = viewer.scene.pick(click.position);

      if (!picked || !picked.id) {
        return;
      }

      const cartesian = viewer.scene.pickPositionSupported
        ? viewer.scene.pickPosition(click.position)
        : viewer.camera.pickEllipsoid(
            click.position,
            viewer.scene.globe.ellipsoid
          );

      if (!cartesian) {
        return;
      }

      const entry = satelliteEntriesRef.current.find(
        (item) => item.entity === picked.id
      );

      if (!entry) {
        return;
      }

      viewer.selectedEntity = entry.entity;
      drawOrbit(entry);
      updateSelectedObject(entry);
    }, Cesium.ScreenSpaceEventType.LEFT_CLICK);

    return () => {
      viewer.clock.onTick.removeEventListener(updateSatellitePositions);

      if (handlerRef.current && !handlerRef.current.isDestroyed()) {
        handlerRef.current.destroy();
        handlerRef.current = null;
      }

      if (orbitEntityRef.current) {
        viewer.entities.remove(orbitEntityRef.current);
        orbitEntityRef.current = null;
      }

      viewer.entities.removeAll();
      satelliteEntriesRef.current = [];

      if (!viewer.isDestroyed()) {
        viewer.destroy();
      }

      viewerRef.current = null;
    };
  }, []);

  useEffect(() => {
    const viewer = viewerRef.current;

    if (!viewer || viewer.isDestroyed()) {
      return;
    }

    if (orbitEntityRef.current) {
      viewer.entities.remove(orbitEntityRef.current);
      orbitEntityRef.current = null;
    }

    if (viewer.selectedEntity) {
      viewer.selectedEntity = undefined;
    }

    viewer.entities.suspendEvents();

    try {
      satelliteEntriesRef.current.forEach(({ entity }) => {
        viewer.entities.remove(entity);
      });
      satelliteEntriesRef.current = [];

      (objects ?? []).forEach((obj) => {
        if (!obj?.line1 || !obj?.line2) {
          return;
        }

        try {
          const satrec = satellite.twoline2satrec(obj.line1, obj.line2);
          const objectType = obj.type === "debris" ? "debris" : "satellite";
          const entity = viewer.entities.add({
            name: obj.name,
            position: Cesium.Cartesian3.fromDegrees(0, 0, 0),
            point: {
              pixelSize: 3,
              color:
                objectType === "debris"
                  ? Cesium.Color.RED
                  : Cesium.Color.LIME,
              disableDepthTestDistance: 0,
            },
          });

          entity.currentAltitudeKm = 0;
          entity.currentVelocityKmS = 0;

          satelliteEntriesRef.current.push({
            entity,
            satrec,
            objectType,
          });
        } catch (error) {
          console.log("TLE error:", error);
        }
      });
    } finally {
      viewer.entities.resumeEvents();
    }

    lastPositionUpdateRef.current = 0;
    viewer.scene.requestRender();
  }, [objects]);

  return (
    <div
      id="cesiumContainer"
      ref={containerRef}
      style={{ width: "100%", height: "100%" }}
    />
  );
};

export default EarthViewer;
